package org.example

/**
 * DllamaProxy.kt
 *
 * Kotlin rewrite of the dllama proxy server logic from src/app.cpp and src/nn/nn-network.cpp.
 *
 * The proxy sits between a root inference node and one or more worker nodes:
 *   root <-> proxy <-> worker[0], worker[1], ..., worker[N-1]
 *
 * Stages:
 *   1. Connect to all workers in isolated mode (each worker sees only the proxy).
 *   2. Accept a root connection; handshake (send nNodes, receive ack).
 *   3. Forward NnNetConfig + NnNodeConfig from root to each worker.
 *   4. Forward weight packets from root to each worker.
 *   5. Relay inference control packets and sync traffic between root and workers.
 *
 * Usage:
 *   kotlin DllamaProxy.kt --workers <host:port> [<host:port>...] [--host <bind>] [--port <port>]
 */

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val SOCKET_BUF_SIZE = 256 * 1024  // 256 kB read/write buffer per socket

// ---------------------------------------------------------------------------
// Constants (must match C++ values in nn-core.hpp / nn-network.cpp)
// ---------------------------------------------------------------------------

private const val ACK: Int = 23571114
private const val OP_MATMUL: Int = 5
private const val SYNC_WITH_ROOT: Int = 0
private const val SYNC_NODE_SLICES: Int = 1
private const val SYNC_NODE_SLICES_EXCEPT_ROOT: Int = 2

// NnSize3D wire layout: floatType(u32) z(u32) y(u32) x(u32) length(u64) nBytes(u64) nBytesXY(u64) = 40 bytes
private const val SIZE3D_BYTES = 40
// NnPointerConfig wire layout: source(u32) pointerIndex(u32) type(u32) = 12 bytes
private const val POINTER_CONFIG_BYTES = 12

// ---------------------------------------------------------------------------
// Socket I/O helpers (little-endian, matching C struct layout on x86)
// ---------------------------------------------------------------------------

// Reusable scratch buffer for small reads/writes — avoids per-call allocation in hot path
private val ioBuf = ThreadLocal.withInitial { ByteArray(8) }

private fun InputStream.readExact(buf: ByteArray, off: Int = 0, len: Int = buf.size) {
    var pos = off; var rem = len
    while (rem > 0) { val n = read(buf, pos, rem); if (n < 0) error("Socket closed"); pos += n; rem -= n }
}

private fun OutputStream.writeExact(buf: ByteArray, off: Int = 0, len: Int = buf.size) { write(buf, off, len) }

private fun InputStream.readInt32LE(): Int {
    val b = ioBuf.get(); readExact(b, 0, 4)
    return (b[0].toInt() and 0xFF) or
           ((b[1].toInt() and 0xFF) shl 8) or
           ((b[2].toInt() and 0xFF) shl 16) or
           ((b[3].toInt() and 0xFF) shl 24)
}

private fun OutputStream.writeInt32LE(v: Int) {
    val b = ioBuf.get()
    b[0] = (v and 0xFF).toByte()
    b[1] = ((v ushr 8) and 0xFF).toByte()
    b[2] = ((v ushr 16) and 0xFF).toByte()
    b[3] = ((v ushr 24) and 0xFF).toByte()
    writeExact(b, 0, 4)
}

private fun InputStream.readInt64LE(): Long {
    val b = ioBuf.get(); readExact(b, 0, 8)
    return (b[0].toLong() and 0xFF) or
           ((b[1].toLong() and 0xFF) shl 8) or
           ((b[2].toLong() and 0xFF) shl 16) or
           ((b[3].toLong() and 0xFF) shl 24) or
           ((b[4].toLong() and 0xFF) shl 32) or
           ((b[5].toLong() and 0xFF) shl 40) or
           ((b[6].toLong() and 0xFF) shl 48) or
           ((b[7].toLong() and 0xFF) shl 56)
}

private fun OutputStream.writeInt64LE(v: Long) {
    val b = ioBuf.get()
    b[0] = (v and 0xFF).toByte()
    b[1] = ((v ushr 8) and 0xFF).toByte()
    b[2] = ((v ushr 16) and 0xFF).toByte()
    b[3] = ((v ushr 24) and 0xFF).toByte()
    b[4] = ((v ushr 32) and 0xFF).toByte()
    b[5] = ((v ushr 40) and 0xFF).toByte()
    b[6] = ((v ushr 48) and 0xFF).toByte()
    b[7] = ((v ushr 56) and 0xFF).toByte()
    writeExact(b, 0, 8)
}

private fun InputStream.readBytes(n: Int): ByteArray { val b = ByteArray(n); readExact(b); return b }
private fun OutputStream.writeBytes(b: ByteArray) = writeExact(b)

private fun InputStream.readAck() { val v = readInt32LE(); if (v != ACK) error("Expected ACK got $v") }
private fun OutputStream.writeAck() = writeInt32LE(ACK)

// NnSize3D wire layout: floatType(u32=0) z(u32=4) y(u32=8) x(u32=12) length(u64=16) nBytes(u64=24) nBytesXY(u64=32)
private fun size3dNBytes(raw: ByteArray): Long =
    ByteBuffer.wrap(raw, 24, 8).order(ByteOrder.LITTLE_ENDIAN).long

private fun size3dY(raw: ByteArray): Long =
    ByteBuffer.wrap(raw, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL

private fun size3dX(raw: ByteArray): Long =
    ByteBuffer.wrap(raw, 12, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL

// ---------------------------------------------------------------------------
// Data classes
// ---------------------------------------------------------------------------

data class SyncInfo(val pipeIndex: Int, val syncType: Int)

data class OpInfo(
    val code: Int,
    val weightSizeNBytes: Long,
    val weightY: Long,   // NnSize3D.y — rows of the weight matrix
    val weightX: Long,   // NnSize3D.x — cols of the weight matrix
    val configSize: Int,
    val config: ByteArray?
)

data class WorkerInfo(
    val nodeIndex: Int,
    val nSegments: Int,
    val nBuffers: Int,
    val ops: List<OpInfo>,
    val segments: List<List<SyncInfo>>,
    var weightBytes: Long = 0L,
    var matmulOpsPerToken: Long = 0L
)

// ---------------------------------------------------------------------------
// SocketConn: socket + its buffered streams (created once, reused everywhere)
// ---------------------------------------------------------------------------

data class SocketConn(
    val socket: Socket,
    val input: InputStream,
    val output: OutputStream
) {
    fun flush() = output.flush()
    fun close() = socket.close()
    val remoteAddress get() = socket.remoteSocketAddress as InetSocketAddress
}

private fun Socket.buffered(): SocketConn = SocketConn(
    this,
    BufferedInputStream(getInputStream(), SOCKET_BUF_SIZE),
    BufferedOutputStream(getOutputStream(), SOCKET_BUF_SIZE)
)

// ---------------------------------------------------------------------------
// Worker connection -- mirrors NnNetwork::connectIsolated
// ---------------------------------------------------------------------------

private fun connectWorkerIsolated(host: String, port: Int, workerIndex: Int, totalNodes: Int): SocketConn {
    // Remap wildcard bind address to loopback for outbound connections
    val connectHost = if (host == "0.0.0.0" || host == "::") "127.0.0.1" else host
    println("   Connecting to worker $workerIndex at $connectHost:$port (isolated)...")
    val sock = Socket()
    sock.connect(InetSocketAddress(connectHost, port), 15_000)
    sock.tcpNoDelay = true
    val conn = sock.buffered()
    // Send: nSockets=1, nNodes=totalNodes, nodeIndex=workerIndex+1 (1-based: root=0, workers=1..N)
    // nPeers = nSockets - 1 = 0, so worker skips peer loop and sends ACK immediately
    conn.output.writeInt32LE(1)                // nSockets for this worker (only the proxy socket)
    conn.output.writeInt32LE(totalNodes)       // total logical nodes (root + all workers)
    conn.output.writeInt32LE(workerIndex + 1)  // 1-based node index (root=0, workers=1..N)
    conn.flush()
    // 0 peers -> worker sends ACK immediately
    conn.input.readAck()
    println("   Worker $workerIndex connected (isolated)")
    return conn
}

private fun signalWorkersReady(workers: List<SocketConn>) {
    for (conn in workers) { conn.output.writeAck(); conn.flush() }
}

// ---------------------------------------------------------------------------
// Stage 3: Forward NnNetConfig + NnNodeConfig for one worker
// ---------------------------------------------------------------------------

private fun forwardConfig(
    workerIdx: Int,
    rootIn: InputStream, rootOut: OutputStream, rootFlush: () -> Unit,
    worker: SocketConn,
    pipeSizesOut: MutableList<ByteArray>  // populated only when workerIdx == 0
): WorkerInfo {
    val wOut = worker.output
    val wIn  = worker.input
    val buf  = ByteArray(65536)
    fun r2w(n: Int) { rootIn.readExact(buf, 0, n); wOut.writeExact(buf, 0, n) }

    // ---- writeNet ----
    // ACK from root -> forward to worker
    rootIn.readAck(); wOut.writeAck(); worker.flush()

    val nBatches = rootIn.readInt32LE(); wOut.writeInt32LE(nBatches)
    val nNodes   = rootIn.readInt32LE(); wOut.writeInt32LE(nNodes)
    val nPipes   = rootIn.readInt32LE(); wOut.writeInt32LE(nPipes)

    for (p in 0 until nPipes) {
        val sz = rootIn.readBytes(SIZE3D_BYTES); wOut.writeBytes(sz)
        if (workerIdx == 0) pipeSizesOut.add(sz)
        val sl = rootIn.readInt32LE(); wOut.writeInt32LE(sl)
        r2w(sl)
    }

    val nPreSyncs = rootIn.readInt32LE(); wOut.writeInt32LE(nPreSyncs)
    for (i in 0 until nPreSyncs) { val pi = rootIn.readInt32LE(); wOut.writeInt32LE(pi) }
    worker.flush()

    // ACK from worker -> forward to root
    wIn.readAck(); rootOut.writeAck(); rootFlush()

    // ---- writeNode ----
    rootIn.readAck(); wOut.writeAck(); worker.flush()

    val nodeIndex = rootIn.readInt32LE(); wOut.writeInt32LE(nodeIndex)
    val nBuffers  = rootIn.readInt32LE(); wOut.writeInt32LE(nBuffers)
    val nSegments = rootIn.readInt32LE(); wOut.writeInt32LE(nSegments)

    for (b in 0 until nBuffers) {
        val sz = rootIn.readBytes(SIZE3D_BYTES); wOut.writeBytes(sz)
        val sl = rootIn.readInt32LE(); wOut.writeInt32LE(sl)
        r2w(sl)
    }

    val allOps      = mutableListOf<OpInfo>()
    val allSegments = mutableListOf<List<SyncInfo>>()

    for (s in 0 until nSegments) {
        val nSyncs = rootIn.readInt32LE(); wOut.writeInt32LE(nSyncs)
        val nOps   = rootIn.readInt32LE(); wOut.writeInt32LE(nOps)

        val segSyncs = mutableListOf<SyncInfo>()
        for (sy in 0 until nSyncs) {
            val pi = rootIn.readInt32LE(); wOut.writeInt32LE(pi)
            val st = rootIn.readInt32LE(); wOut.writeInt32LE(st)
            if (workerIdx == 0) segSyncs.add(SyncInfo(pi, st))
        }
        if (workerIdx == 0) allSegments.add(segSyncs)

        for (o in 0 until nOps) {
            val code  = rootIn.readInt32LE(); wOut.writeInt32LE(code)
            val idx   = rootIn.readInt32LE(); wOut.writeInt32LE(idx)
            val wsz   = rootIn.readBytes(SIZE3D_BYTES); wOut.writeBytes(wsz)
            val cfgSz = rootIn.readInt32LE(); wOut.writeInt32LE(cfgSz)
            val sl    = rootIn.readInt32LE(); wOut.writeInt32LE(sl); r2w(sl) // op name
            val inp2  = rootIn.readBytes(POINTER_CONFIG_BYTES); wOut.writeBytes(inp2)
            val outp  = rootIn.readBytes(POINTER_CONFIG_BYTES); wOut.writeBytes(outp)
            val cfg: ByteArray? = if (cfgSz > 0) { val c = rootIn.readBytes(cfgSz); wOut.writeBytes(c); c } else null
            allOps.add(OpInfo(code, size3dNBytes(wsz), size3dY(wsz), size3dX(wsz), cfgSz, cfg))
        }
    }
    worker.flush()

    wIn.readAck(); rootOut.writeAck(); rootFlush()

    // Compute stats from ops — mirrors the C++ proxy's computeNodeMatmulOps lambda in app.cpp
    var weightBytes = 0L
    var matmulOps   = 0L
    for (op in allOps) {
        weightBytes += op.weightSizeNBytes
        if (op.code == OP_MATMUL && op.config != null && op.configSize >= 8) {
            // NnMatmulOpConfig: nExperts(u32) nActiveExperts(u32) activeExpertIndexesBufferIndex(u32)
            val bb = ByteBuffer.wrap(op.config).order(ByteOrder.LITTLE_ENDIAN)
            val nExperts = bb.int
            val nActive  = bb.int
            val activeFactor = if (nExperts > 0) nActive.toLong() else 1L
            // GFLOPs/token = 2 * activeFactor * y * x  (matches C++: af * weightSize.y * weightSize.x)
            matmulOps += activeFactor * op.weightY * op.weightX
        }
    }

    return WorkerInfo(nodeIndex, nSegments, nBuffers, allOps, allSegments, weightBytes, matmulOps)
}

// ---------------------------------------------------------------------------
// Stage 4: Forward weight packets for one worker
// ---------------------------------------------------------------------------

private fun forwardWeights(
    workerIdx: Int, workerHost: String, workerPort: Int,
    rootIn: InputStream, rootOut: OutputStream, rootFlush: () -> Unit,
    worker: SocketConn
): Long {
    val wOut = worker.output
    val wIn  = worker.input
    val buf  = ByteArray(1024 * 1024)
    var totalBytes = 0L
    var nPkts = 0
    println("   Forwarding weights to worker $workerIdx ($workerHost:$workerPort)...")

    while (true) {
        val nameSize = rootIn.readInt32LE()
        wOut.writeInt32LE(nameSize)
        if (nameSize == 0) {
            worker.flush()
            // End sentinel: worker sends ACK, proxy forwards it back to root (matches ackW2R in C++)
            val ack = wIn.readInt32LE()
            rootOut.writeInt32LE(ack); rootFlush()
            break
        }
        // op name
        val nameBytes = if (buf.size >= nameSize) buf else ByteArray(nameSize)
        rootIn.readExact(nameBytes, 0, nameSize)
        wOut.writeExact(nameBytes, 0, nameSize)
        val opName = String(nameBytes, 0, nameSize - 1) // null-terminated

        val opIdx  = rootIn.readInt32LE(); wOut.writeInt32LE(opIdx)
        val offset = rootIn.readInt64LE(); wOut.writeInt64LE(offset)
        val nBytes = rootIn.readInt64LE(); wOut.writeInt64LE(nBytes)

        // Forward weight data in chunks
        var rem = nBytes
        while (rem > 0) {
            val chunk = minOf(rem, buf.size.toLong()).toInt()
            rootIn.readExact(buf, 0, chunk)
            wOut.writeExact(buf, 0, chunk)
            rem -= chunk
        }
        worker.flush()
        totalBytes += nBytes
        nPkts++
        println("   Worker[$workerIdx] op=%-22s idx=%3d offset=%8d size=%8d kB".format(
            opName, opIdx, offset, nBytes / 1024))
    }
    println("   Worker[$workerIdx]: $nPkts packets, %.2f MB forwarded".format(totalBytes / (1024.0 * 1024.0)))
    return totalBytes
}

// ---------------------------------------------------------------------------
// LlmControlPacket (8 bytes: position(u32) batchSize(u32))
// ---------------------------------------------------------------------------

private data class LlmControlPacket(val position: Int, val batchSize: Int)

private fun InputStream.readControlPacket(): LlmControlPacket {
    val position  = readInt32LE()
    val batchSize = readInt32LE()
    return LlmControlPacket(position, batchSize)
}

private fun OutputStream.writeControlPacket(pkt: LlmControlPacket) {
    writeInt32LE(pkt.position)
    writeInt32LE(pkt.batchSize)
}


// ---------------------------------------------------------------------------
// Stage 5: Inference relay loop
// ---------------------------------------------------------------------------

private fun inferenceRelayLoop(
    rootIn: InputStream, rootOut: OutputStream, rootFlush: () -> Unit,
    workers: List<SocketConn>,
    syncList: List<SyncInfo>,
    pipeSizes: List<ByteArray>,
    nNodes: Int,
    workerInfos: List<WorkerInfo>
) {
    val nWorkers = workers.size
    var stepIndex = 0

    // Matches C++ getBytes(floatType, x): F_32->4x, F_16->2x, Q40->x/32*18, Q80->x/32*34
    fun pipeTokenBytes(pipeIndex: Int): Int {
        val sz = pipeSizes[pipeIndex]
        val bb = ByteBuffer.wrap(sz).order(ByteOrder.LITTLE_ENDIAN)
        val floatType = bb.int  // NnSize3D.floatType
        bb.int                  // z (skip)
        bb.int                  // y (skip)
        val x = bb.int.toLong() and 0xFFFFFFFFL
        return when (floatType) {
            0    -> (4L * x).toInt()         // F_32
            1    -> (2L * x).toInt()         // F_16
            2    -> (x / 32L * 18L).toInt()  // F_Q40: blockSize=32, blockBytes=18
            3    -> (x / 32L * 34L).toInt()  // F_Q80: blockSize=32, blockBytes=34
            else -> (4L * x).toInt()
        }
    }

    println("Entering inference relay loop...")

    var allSlicesBuf = ByteArray(0)

    while (true) {
        val pkt = rootIn.readControlPacket()

        if (pkt.batchSize == 0) {
            println("Stop signal received, forwarding to all workers")
            for (w in workers) { w.output.writeControlPacket(pkt); w.flush() }
            break
        }

        println("Step $stepIndex: position=${pkt.position} batchSize=${pkt.batchSize}")
        for (w in 0 until nWorkers) {
            val addr = workers[w].remoteAddress
            println("   Worker[$w] (${addr.hostString}:${addr.port}): ${"%.3f".format(workerInfos[w].matmulOpsPerToken * 2.0 * pkt.batchSize / 1e9)} GFLOPs")
        }

        // Broadcast control packet to all workers
        for (w in workers) { w.output.writeControlPacket(pkt); w.flush() }

        val batchSize = pkt.batchSize
        var rootToWorkers = 0L
        var workersToRoot = 0L

        for (sync in syncList) {
            val tokenBytes      = pipeTokenBytes(sync.pipeIndex)
            val sliceTokenBytes = tokenBytes / nNodes

            for (batchIndex in 0 until batchSize) {
                when (sync.syncType) {

                    SYNC_WITH_ROOT -> {
                        // Root sends full pipe; proxy broadcasts to all workers.
                        if (allSlicesBuf.size < tokenBytes) allSlicesBuf = ByteArray(tokenBytes)
                        rootIn.readExact(allSlicesBuf, 0, tokenBytes)
                        for (w in workers) w.output.writeExact(allSlicesBuf, 0, tokenBytes)
                        // Flush all worker outputs together after writing to all
                        for (w in workers) w.flush()
                        rootToWorkers += tokenBytes.toLong() * nWorkers
                    }

                    SYNC_NODE_SLICES -> {
                        // Matches C++ proxy SYNC_NODE_SLICES (lines 687-713 in app.cpp):
                        //   Read root's slice + all worker slices
                        //   Send all worker slices to root
                        //   Send all other nodes' slices to each worker (nNodes-1 slices each)
                        val totalBytes = sliceTokenBytes * nNodes
                        if (allSlicesBuf.size < totalBytes) allSlicesBuf = ByteArray(totalBytes)

                        // Read root's slice then each worker's slice
                        rootIn.readExact(allSlicesBuf, 0, sliceTokenBytes)
                        for (w in 0 until nWorkers)
                            workers[w].input.readExact(allSlicesBuf, (w + 1) * sliceTokenBytes, sliceTokenBytes)

                        // Send all worker slices to root (nWorkers slices)
                        for (n in 1 until nNodes)
                            rootOut.writeExact(allSlicesBuf, n * sliceTokenBytes, sliceTokenBytes)
                        rootFlush()

                        // Send each worker exactly 1 slice at its sliceIndex.
                        // In isolated mode: nSockets=1, socketIndex=0, nodeIndex=w+1
                        //   sliceIndex = if (socketIndex >= nodeIndex) socketIndex+1 else socketIndex
                        //              = if (0 >= w+1) 1 else 0 = 0 for all workers
                        // This matches syncNodeSlices which reads nSocketsPerThread=1 slice.
                        for (w in 0 until nWorkers) {
                            val nodeIndex = w + 1
                            val sliceIndex = if (0 >= nodeIndex) 1 else 0  // always 0
                            workers[w].output.writeExact(allSlicesBuf, sliceIndex * sliceTokenBytes, sliceTokenBytes)
                        }
                        for (w in workers) w.flush()

                        rootToWorkers += sliceTokenBytes.toLong() * nWorkers
                        workersToRoot += sliceTokenBytes.toLong() * nWorkers
                    }

                    SYNC_NODE_SLICES_EXCEPT_ROOT -> {
                        // Workers write mySlice to proxy, do NOT read back (onlyFromWorkerToRoot).
                        // Root reads nNodes-1 slices from proxy.
                        val need = sliceTokenBytes * nWorkers
                        if (allSlicesBuf.size < need) allSlicesBuf = ByteArray(need)

                        for (w in 0 until nWorkers)
                            workers[w].input.readExact(allSlicesBuf, w * sliceTokenBytes, sliceTokenBytes)
                        for (w in 0 until nWorkers)
                            rootOut.writeExact(allSlicesBuf, w * sliceTokenBytes, sliceTokenBytes)
                        rootFlush()
                        workersToRoot += sliceTokenBytes.toLong() * nWorkers
                    }
                }
            }
        }

        println("   Step $stepIndex done: root->workers=${rootToWorkers/1024} kB, workers->root=${workersToRoot/1024} kB")
        stepIndex++
    }

    println("Inference session complete after $stepIndex steps")
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

data class ProxyArgs(
    val workerAddrs: List<Pair<String, Int>>,
    val bindHost: String,
    val bindPort: Int
)

private fun parseArgs(args: Array<String>): ProxyArgs {
    var bindHost = "0.0.0.0"
    var bindPort = 9990
    val workers  = mutableListOf<Pair<String, Int>>()
    var i = 0
    while (i < args.size) {
        when (args[i]) {
            "--workers" -> {
                i++
                while (i < args.size && !args[i].startsWith("-")) {
                    val parts = args[i].split(":")
                    require(parts.size == 2) { "Worker address must be host:port, got: ${args[i]}" }
                    workers.add(parts[0] to parts[1].toInt())
                    i++
                }
                continue
            }
            "--host" -> { bindHost = args[++i] }
            "--port" -> { bindPort = args[++i].toInt() }
            else -> error("Unknown option: ${args[i]}")
        }
        i++
    }
    require(workers.isNotEmpty()) { "Proxy requires at least one worker (--workers host:port ...)" }
    return ProxyArgs(workers, bindHost, bindPort)
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

fun main(args: Array<String>) {
    val cfg      = parseArgs(args)
    val nWorkers = cfg.workerAddrs.size
    val nNodes   = nWorkers + 1 // root + workers

    println("Proxy starting on ${cfg.bindHost}:${cfg.bindPort}, forwarding to $nWorkers worker(s)")

    // -------------------------------------------------------------------------
    // Stage 1: Connect to all workers in isolated mode
    // -------------------------------------------------------------------------
    println("\n[Stage 1] Connecting to $nWorkers worker(s) in isolated mode...")
    val workerSockets = cfg.workerAddrs.mapIndexed { idx, (host, port) ->
        connectWorkerIsolated(host, port, idx, nNodes)
    }
    signalWorkersReady(workerSockets)
    println("[Stage 1] All $nWorkers worker(s) connected\n")

    // Keep accepting root connections indefinitely (like worker mode)
    while (true) {
        // -------------------------------------------------------------------------
        // Stage 2: Accept root connection and handshake
        // -------------------------------------------------------------------------
        val server = ServerSocket()
        server.reuseAddress = true
        server.bind(InetSocketAddress(cfg.bindHost, cfg.bindPort))
        println("Waiting for root node to connect on ${cfg.bindHost}:${cfg.bindPort}...")
        val rootSock = server.accept()
        server.close()
        rootSock.tcpNoDelay = true
        println("Root node connected")

        val root = rootSock.buffered()
        val rootFlush: () -> Unit = { root.flush() }

        // Send nNodes to root, read ack
        root.output.writeInt32LE(nNodes)
        root.flush()
        root.input.readAck()
        println("Handshake complete with root, nNodes=$nNodes\n")

        // -------------------------------------------------------------------------
        // Stage 3: Forward NnNetConfig + NnNodeConfig root -> each worker
        // -------------------------------------------------------------------------
        println("[Stage 3] Forwarding config from root to $nWorkers worker(s)...")
        val pipeSizes   = mutableListOf<ByteArray>()
        val workerInfos = workerSockets.mapIndexed { w, workerConn ->
            val info = forwardConfig(w, root.input, root.output, rootFlush, workerConn, pipeSizes)
            println("   Config forwarded for worker[$w] (nodeIndex=${info.nodeIndex}, nSegments=${info.nSegments})")
            info
        }
        println()

        // Log computation per worker — mirrors C++ proxy's printf in app.cpp
        println("[Stage 3] Computation per worker:")
        for (w in 0 until nWorkers) {
            val wi = workerInfos[w]
            println("   Worker[$w] (${cfg.workerAddrs[w].first}:${cfg.workerAddrs[w].second}): " +
                "weights=${"%.2f".format(wi.weightBytes / (1024.0 * 1024.0))} MB, " +
                "matmul=${"%.3f".format(wi.matmulOpsPerToken * 2.0 / 1e9)} GFLOPs/token")
        }
        println()

        // -------------------------------------------------------------------------
        // Stage 4: Forward weight packets root -> each worker
        // -------------------------------------------------------------------------
        println("[Stage 4] Forwarding weights from root to $nWorkers worker(s)...")
        workerSockets.forEachIndexed { w, workerConn ->
            forwardWeights(w, cfg.workerAddrs[w].first, cfg.workerAddrs[w].second,
                root.input, root.output, rootFlush, workerConn)
        }
        println("[Stage 4] All weights forwarded\n")

        // -------------------------------------------------------------------------
        // Stage 5: Inference relay loop
        // -------------------------------------------------------------------------
        val syncList = workerInfos.firstOrNull()?.segments?.flatten() ?: emptyList()

        try {
            inferenceRelayLoop(root.input, root.output, rootFlush, workerSockets, syncList, pipeSizes, nNodes, workerInfos)
        } catch (e: Exception) {
            println("Network error during inference: ${e.message}")
        } finally {
            root.close()
        }
    }
}
