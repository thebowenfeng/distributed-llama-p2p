package org.example

/**
 * DllamaProxy.kt
 *
 * A distributed inference proxy server that coordinates computation between a root node and multiple worker nodes.
 *
 * ARCHITECTURE OVERVIEW:
 * This is a Kotlin rewrite of the dllama proxy server (runProxyApp in src/app.cpp), matching the C++ implementation exactly.
 * The proxy acts as a relay/orchestrator in a distributed neural network inference setup:
 *
 * 1. ROOT NODE connects to the PROXY and sends:
 *    - Model configuration (network topology, layer operations, synchronization points)
 *    - Model weights (weight tensors for each layer operation)
 *    - Inference batches (token positions and batch sizes to process)
 *
 * 2. PROXY connects to multiple WORKER NODES and:
 *    - Forwards configuration in parallel to all workers
 *    - Forwards weights to workers in a round-robin fashion
 *    - Orchestrates synchronization during inference (data exchanges between workers)
 *
 * 3. WORKER NODES execute local computation and synchronize via the proxy
 *
 * PROTOCOL FLOW (5 Stages):
 *   [Stage 1] Connect to all workers in isolated mode
 *   [Stage 2] Forward model configuration from root to workers
 *   [Stage 3] Forward model weights from root to workers (round-robin)
 *   [Stage 4] Enter inference relay loop for each batch
 *   [Stage 5] Complete inference session
 *
 * KEY SYNCHRONIZATION MECHANISMS:
 *   - ACK values: Handshake verification between nodes
 *   - ControlPackets: Position + batch size for each inference step
 *   - SyncInfo: Specifies which pipe and synchronization type for data exchange
 *   - Pipe slices: Different synchronization strategies (with root, between workers, etc.)
 *
 * Usage:
 *   kotlin DllamaProxy.kt --workers <host:port> [<host:port>...] [--host <bind>] [--port <port>]
 *
 * Example:
 *   kotlin DllamaProxy.kt --workers worker1.local:9001 worker2.local:9001 --host 0.0.0.0 --port 9990
 */

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder

// ============================================================================
// PROTOCOL CONSTANTS
// ============================================================================

/** Magic number used for ACK/handshake verification between nodes */
private const val ACK: Int = 23571114

/** Operation code for matrix multiplication */
private const val OP_MATMUL: Int = 5

/** Size in bytes of a Size3D structure (float_type=4, pad=4, y=4, x=4, stride_y=4, stride_x=4, nbytes=8) */
private const val SIZE3D_BYTES: Int = 40

/** Size in bytes for pointer configuration data in an operation */
private const val PTR_CFG_BYTES: Int = 12

/** Buffer size for buffered I/O streams (256 KB) */
private const val BUF_SIZE: Int = 256 * 1024

// Synchronization type constants for pipe slices
/** Synchronize data with the root node (root sends data to all workers) */
private const val SYNC_WITH_ROOT: Int = 0

/** Synchronize slices across all nodes (each node contributes a slice) */
private const val SYNC_NODE_SLICES: Int = 1

/** Synchronize slices between workers only (excluding root node) */
private const val SYNC_NODE_SLICES_EXCEPT_ROOT: Int = 2

// ============================================================================
// THREAD-LOCAL I/O BUFFER
// ============================================================================

/**
 * Thread-local buffer for efficient integer/long serialization.
 * Reused across multiple I/O operations to avoid repeated allocation.
 */
private val ioBuf = ThreadLocal.withInitial { ByteArray(8) }

// ============================================================================
// EXTENSION FUNCTIONS: EXACT I/O
// ============================================================================

/**
 * Reads exactly [len] bytes from the input stream into [buf] starting at offset [off].
 * Blocks until all bytes are read or socket is closed.
 *
 * @param buf The byte array to read into
 * @param off Starting offset in the buffer (default: 0)
 * @param len Number of bytes to read (default: full buffer from offset)
 * @throws EOFException if the socket is closed before all bytes are read
 */
private fun InputStream.readExact(buf: ByteArray, off: Int = 0, len: Int = buf.size) {
    var pos = off
    var rem = len
    while (rem > 0) {
        val bytesRead = read(buf, pos, rem)
        if (bytesRead < 0) throw EOFException("Socket closed")
        pos += bytesRead
        rem -= bytesRead
    }
}

/**
 * Writes exactly [len] bytes from [buf] starting at offset [off].
 *
 * @param buf The byte array to write from
 * @param off Starting offset in the buffer (default: 0)
 * @param len Number of bytes to write (default: full buffer from offset)
 */
private fun OutputStream.writeExact(buf: ByteArray, off: Int = 0, len: Int = buf.size) = write(buf, off, len)

// ============================================================================
// EXTENSION FUNCTIONS: 32-BIT INTEGER SERIALIZATION (LITTLE-ENDIAN)
// ============================================================================

/**
 * Reads a 32-bit signed integer in little-endian format from the input stream.
 * Uses thread-local buffer for efficiency.
 */
private fun InputStream.readI32(): Int {
    val scratchBuf = ioBuf.get()
    readExact(scratchBuf, 0, 4)
    return (scratchBuf[0].toInt() and 0xFF) or
           ((scratchBuf[1].toInt() and 0xFF) shl 8) or
           ((scratchBuf[2].toInt() and 0xFF) shl 16) or
           ((scratchBuf[3].toInt() and 0xFF) shl 24)
}

/**
 * Writes a 32-bit signed integer in little-endian format to the output stream.
 * Uses thread-local buffer for efficiency.
 */
private fun OutputStream.writeI32(value: Int) {
    val scratchBuf = ioBuf.get()
    scratchBuf[0] = (value and 0xFF).toByte()
    scratchBuf[1] = ((value ushr 8) and 0xFF).toByte()
    scratchBuf[2] = ((value ushr 16) and 0xFF).toByte()
    scratchBuf[3] = ((value ushr 24) and 0xFF).toByte()
    writeExact(scratchBuf, 0, 4)
}

// ============================================================================
// EXTENSION FUNCTIONS: 64-BIT INTEGER SERIALIZATION (LITTLE-ENDIAN)
// ============================================================================

/**
 * Reads a 64-bit signed integer in little-endian format from the input stream.
 * Uses thread-local buffer for efficiency.
 */
private fun InputStream.readI64(): Long {
    val scratchBuf = ioBuf.get()
    readExact(scratchBuf, 0, 8)
    return (scratchBuf[0].toLong() and 0xFF) or
           ((scratchBuf[1].toLong() and 0xFF) shl 8) or
           ((scratchBuf[2].toLong() and 0xFF) shl 16) or
           ((scratchBuf[3].toLong() and 0xFF) shl 24) or
           ((scratchBuf[4].toLong() and 0xFF) shl 32) or
           ((scratchBuf[5].toLong() and 0xFF) shl 40) or
           ((scratchBuf[6].toLong() and 0xFF) shl 48) or
           ((scratchBuf[7].toLong() and 0xFF) shl 56)
}

/**
 * Writes a 64-bit signed integer in little-endian format to the output stream.
 * Uses thread-local buffer for efficiency.
 */
private fun OutputStream.writeI64(value: Long) {
    val scratchBuf = ioBuf.get()
    scratchBuf[0] = (value and 0xFF).toByte()
    scratchBuf[1] = ((value ushr 8) and 0xFF).toByte()
    scratchBuf[2] = ((value ushr 16) and 0xFF).toByte()
    scratchBuf[3] = ((value ushr 24) and 0xFF).toByte()
    scratchBuf[4] = ((value ushr 32) and 0xFF).toByte()
    scratchBuf[5] = ((value ushr 40) and 0xFF).toByte()
    scratchBuf[6] = ((value ushr 48) and 0xFF).toByte()
    scratchBuf[7] = ((value ushr 56) and 0xFF).toByte()
    writeExact(scratchBuf, 0, 8)
}

// ============================================================================
// EXTENSION FUNCTIONS: ACK PROTOCOL
// ============================================================================

/**
 * Reads and verifies an ACK value from the input stream.
 * ACKs are used as handshakes to synchronize communication between nodes.
 *
 * @throws IllegalStateException if the received value doesn't match the expected ACK constant
 */
private fun InputStream.readAck() {
    val ackValue = readI32()
    if (ackValue != ACK) error("ACK mismatch: got $ackValue")
}

/**
 * Writes an ACK value to the output stream.
 * ACKs are used as handshakes to synchronize communication between nodes.
 */
private fun OutputStream.writeAck() = writeI32(ACK)

// ============================================================================
// EXTENSION FUNCTIONS: BULK DATA
// ============================================================================

/**
 * Reads exactly [n] bytes from the input stream and returns them as a ByteArray.
 */
private fun InputStream.readBytes(n: Int): ByteArray {
    val buf = ByteArray(n)
    readExact(buf)
    return buf
}

// ============================================================================
// EXTENSION FUNCTIONS: SIZE3D STRUCTURE ACCESSORS
// ============================================================================

/**
 * Extracts the float type field from a Size3D structure.
 * Float type: 0=f32, 1=f16, 2=q4_0, 3=q8_0, etc.
 */
private fun ByteArray.size3dFloatType(): Int =
    ByteBuffer.wrap(this, 0, 4).order(ByteOrder.LITTLE_ENDIAN).int

/**
 * Extracts the Y dimension (height) from a Size3D structure.
 */
private fun ByteArray.size3dY(): Long =
    (ByteBuffer.wrap(this, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL)

/**
 * Extracts the X dimension (width) from a Size3D structure.
 */
private fun ByteArray.size3dX(): Long =
    (ByteBuffer.wrap(this, 12, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL)

/**
 * Extracts the total size in bytes from a Size3D structure.
 */
private fun ByteArray.size3dNBytes(): Long =
    ByteBuffer.wrap(this, 24, 8).order(ByteOrder.LITTLE_ENDIAN).long

// ============================================================================
// PIPE TOKEN SIZE CALCULATION
// ============================================================================

/**
 * Calculates the number of bytes required to represent one token for a given pipe.
 * The size depends on the data type and X dimension of the tensor.
 *
 * @param sizeBlob A Size3D structure containing the tensor metadata
 * @return Number of bytes per token for this pipe
 */
private fun pipeTokenBytes(sizeBlob: ByteArray): Int {
    val tensorWidth = sizeBlob.size3dX()
    return when (sizeBlob.size3dFloatType()) {
        0 -> (4L * tensorWidth).toInt()      // f32: 4 bytes per element
        1 -> (2L * tensorWidth).toInt()      // f16: 2 bytes per element
        2 -> (tensorWidth / 32L * 18L).toInt() // q4_0: 18 bytes per 32 elements
        3 -> (tensorWidth / 32L * 34L).toInt() // q8_0: 34 bytes per 32 elements
        else -> (4L * tensorWidth).toInt()   // default to f32
    }
}

// ============================================================================
// DATA CLASSES: SYNCHRONIZATION INFO
// ============================================================================

/**
 * Describes a single synchronization point within a segment.
 *
 * @param pipeIndex Index of the pipe (output tensor) that needs synchronization
 * @param syncType Type of synchronization: SYNC_WITH_ROOT, SYNC_NODE_SLICES, or SYNC_NODE_SLICES_EXCEPT_ROOT
 */
private data class SyncInfo(val pipeIndex: Int, val syncType: Int)

// ============================================================================
// DATA CLASSES: OPERATION INFO
// ============================================================================

/**
 * Contains metadata about a single operation (e.g., matrix multiplication) in the computation graph.
 *
 * @param code Operation type code (e.g., OP_MATMUL = 5)
 * @param weightSizeNBytes Total size of weight tensors for this operation in bytes
 * @param weightY Y dimension (height) of weight tensors
 * @param weightX X dimension (width) of weight tensors
 * @param configSize Size of operation-specific configuration data in bytes
 * @param config Optional configuration blob (e.g., for MoE: number of experts and active experts)
 */
private data class OpInfo(
    val code: Int,
    val weightSizeNBytes: Long,
    val weightY: Long,
    val weightX: Long,
    val configSize: Int,
    val config: ByteArray?
)

// ============================================================================
// DATA CLASSES: WORKER INFO
// ============================================================================

/**
 * Contains configuration and metadata for a single worker node.
 * Populated during [forwardConfig] and used for logging and inference coordination.
 *
 * @param nodeIndex Unique index assigned to this worker in the distributed graph
 * @param nSegments Number of computation segments (layers/stages) this worker participates in
 * @param nBuffers Number of buffer allocations for intermediate activations
 * @param ops List of operations this worker must execute
 * @param segments List of synchronization points organized by segment
 * @param weightBytes Total size of weight tensors for all operations
 * @param matmulOpsPerToken Estimated FLOPs per token for all matrix multiplications (used for performance logging)
 */
private data class WorkerInfo(
    val nodeIndex: Int,
    val nSegments: Int,
    val nBuffers: Int,
    val ops: List<OpInfo>,
    val segments: List<List<SyncInfo>>,
    val weightBytes: Long,
    val matmulOpsPerToken: Long
)

// ============================================================================
// DATA CLASSES: SOCKET CONNECTION
// ============================================================================

/**
 * Wraps a socket connection with buffered input and output streams.
 * Simplifies I/O operations and provides convenience methods.
 *
 * @param socket The underlying socket
 * @param inp Buffered input stream for reading
 * @param out Buffered output stream for writing
 */
private data class SocketConn(
    val socket: Socket,
    val inp: InputStream,
    val out: OutputStream
) {
    /** Flushes the output buffer */
    fun flush() = out.flush()

    /** Closes the socket safely */
    fun close() = runCatching { socket.close() }

    /** Gets the remote address of the connected peer */
    val remoteAddr: InetSocketAddress get() = socket.remoteSocketAddress as InetSocketAddress
}

/**
 * Wraps a raw socket with buffered streams for efficient I/O.
 *
 * @return A SocketConn with buffered input and output
 */
private fun Socket.buffered() = SocketConn(
    this,
    BufferedInputStream(getInputStream(), BUF_SIZE),
    BufferedOutputStream(getOutputStream(), BUF_SIZE)
)

// ============================================================================
// DATA CLASSES: CONTROL PACKET
// ============================================================================

/**
 * Represents a control packet sent at the start of each inference step.
 * Tells workers which tokens to process and how many to batch together.
 *
 * @param position Starting token position for this inference step
 * @param batchSize Number of tokens to process in this batch (0 signals end of inference)
 */
private data class ControlPacket(val position: Int, val batchSize: Int)

/**
 * Reads a control packet from the input stream.
 */
private fun InputStream.readControlPacket() = ControlPacket(readI32(), readI32())

/**
 * Writes a control packet to the output stream.
 */
private fun OutputStream.writeControlPacket(controlPacket: ControlPacket) {
    writeI32(controlPacket.position)
    writeI32(controlPacket.batchSize)
}

// ============================================================================
// WORKER CONNECTION: ISOLATED MODE
// ============================================================================

/**
 * Establishes connections to all worker nodes in "isolated mode".
 *
 * PROTOCOL:
 * 1. For each worker, open a TCP connection and establish handshake:
 *    - Send isolated_mode=1, totalNodes, nodeIndex
 *    - Receive ACK from worker
 * 2. After all workers have individually acknowledged, send ACK to each
 *
 * Isolated mode means each worker operates as a separate node in the distributed graph.
 * This is in contrast to a pipeline mode where workers would form a chain.
 *
 * @param workerAddrs List of (host, port) pairs for each worker
 * @return List of established SocketConn objects for each worker, in order
 */
private fun connectWorkersIsolated(workerAddrs: List<Pair<String, Int>>): List<SocketConn> {
    val nWorkers = workerAddrs.size
    val totalNodes = nWorkers + 1  // workers + root node
    val conns = mutableListOf<SocketConn>()

    for ((workerIndex, addr) in workerAddrs.withIndex()) {
        val (host, port) = addr
        // Convert wildcard addresses to loopback for local connections
        val connectHost = if (host == "0.0.0.0" || host == "::") "127.0.0.1" else host
        println("   Connecting to worker $workerIndex at $connectHost:$port (isolated)...")

        val socket = Socket().also {
            it.connect(InetSocketAddress(connectHost, port), 15_000)
            it.tcpNoDelay = true  // Disable Nagle for lower latency
        }
        val workerConn = socket.buffered()

        // Send isolated mode handshake: mode=1, totalNodes, nodeIndex
        workerConn.out.writeI32(1)                    // isolated_mode = 1
        workerConn.out.writeI32(totalNodes)           // total number of nodes in graph
        workerConn.out.writeI32(workerIndex + 1)      // node index for this worker (1-indexed)
        workerConn.flush()

        // Wait for worker to acknowledge the handshake
        workerConn.inp.readAck()
        conns.add(workerConn)
        println("   Worker $workerIndex connected (isolated)")
    }

    // Final synchronization: proxy acknowledges to all workers
    for (workerConn in conns) {
        workerConn.out.writeAck()
        workerConn.flush()
    }
    return conns
}

// ============================================================================
// CONFIG FORWARDING: ROOT TO WORKERS
// ============================================================================

/**
 * Forwards model configuration from the root node to all workers.
 *
 * CONFIGURATION INCLUDES:
 * - Batch size, number of nodes, number of pipes (output tensors)
 * - For each pipe: size metadata and slice information
 * - Synchronization topology
 * - For each buffer: size metadata and slice information
 * - For each operation: code, weight size, configuration, and pointers
 *
 * PROTOCOL:
 * Multiple rounds of handshaking (ACK exchanges) to ensure synchronization.
 *
 * @param rootIn Input stream from root node
 * @param rootOut Output stream to root node
 * @param rootFlush Lambda to flush root output stream
 * @param workers List of connected worker socket connections
 * @param pipeSizesOut Mutable list to collect pipe size metadata (populated by this function)
 * @return List of WorkerInfo objects containing parsed configuration for each worker
 */
private fun forwardConfig(
    rootIn: InputStream,
    rootOut: OutputStream,
    rootFlush: () -> Unit,
    workers: List<SocketConn>,
    pipeSizesOut: MutableList<ByteArray>
): List<WorkerInfo> {
    val scratchBuf = ByteArray(65536)
    var netNNodes = 0

    /**
     * Helper: Reads [bytesCount] bytes from root and forwards to worker output stream.
     */
    fun forwardRootToWorker(workerOut: OutputStream, bytesCount: Int) {
        val buf = if (scratchBuf.size >= bytesCount) scratchBuf else ByteArray(bytesCount)
        rootIn.readExact(buf, 0, bytesCount)
        workerOut.writeExact(buf, 0, bytesCount)
    }

    return workers.mapIndexed { workerIndex, workerConn ->
        val workerOut = workerConn.out
        val workerIn = workerConn.inp

        // Handshake round 1: acknowledge config is coming
        rootIn.readAck()
        workerOut.writeAck()
        workerConn.flush()

        // Read and forward basic topology
        val nBatches = rootIn.readI32()
        workerOut.writeI32(nBatches)
        val nNodes = rootIn.readI32()
        workerOut.writeI32(nNodes)
        if (workerIndex == 0) netNNodes = nNodes  // Store from first worker

        val nPipes = rootIn.readI32()
        workerOut.writeI32(nPipes)

        // Read and forward pipe metadata
        for (pipeIdx in 0 until nPipes) {
            val sizeBlob = rootIn.readBytes(SIZE3D_BYTES)
            workerOut.writeExact(sizeBlob)
            if (workerIndex == 0) pipeSizesOut.add(sizeBlob)

            val sliceLength = rootIn.readI32()
            workerOut.writeI32(sliceLength)
            forwardRootToWorker(workerOut, sliceLength)
        }

        // Read and forward pre-sync topology
        val nPreSyncs = rootIn.readI32()
        workerOut.writeI32(nPreSyncs)
        for (i in 0 until nPreSyncs) {
            val pipeIndex = rootIn.readI32()
            workerOut.writeI32(pipeIndex)
        }
        workerConn.flush()

        // Handshake round 2: worker acknowledges receipt
        workerIn.readAck()
        rootOut.writeAck()
        rootFlush()

        // Handshake round 3: send buffer configuration
        rootIn.readAck()
        workerOut.writeAck()
        workerConn.flush()

        // Read and forward buffer metadata
        val nodeIndex = rootIn.readI32()
        workerOut.writeI32(nodeIndex)
        val nBuffers = rootIn.readI32()
        workerOut.writeI32(nBuffers)
        val nSegments = rootIn.readI32()
        workerOut.writeI32(nSegments)

        for (bufIdx in 0 until nBuffers) {
            val sizeBlob = rootIn.readBytes(SIZE3D_BYTES)
            workerOut.writeExact(sizeBlob)

            val sliceLength = rootIn.readI32()
            workerOut.writeI32(sliceLength)
            forwardRootToWorker(workerOut, sliceLength)
        }

        // Read and forward operation metadata
        val allOps = mutableListOf<OpInfo>()
        val allSegments = mutableListOf<List<SyncInfo>>()

        for (segmentIdx in 0 until nSegments) {
            val nSyncs = rootIn.readI32()
            workerOut.writeI32(nSyncs)
            val nOps = rootIn.readI32()
            workerOut.writeI32(nOps)

            // Read sync points for this segment
            val segmentSyncs = mutableListOf<SyncInfo>()
            for (syncIdx in 0 until nSyncs) {
                val pipeIndex = rootIn.readI32()
                workerOut.writeI32(pipeIndex)
                val syncType = rootIn.readI32()
                workerOut.writeI32(syncType)
                segmentSyncs.add(SyncInfo(pipeIndex, syncType))
            }
            allSegments.add(segmentSyncs)

            // Read operations for this segment
            for (opIdx in 0 until nOps) {
                val code = rootIn.readI32()
                workerOut.writeI32(code)

                val index = rootIn.readI32()
                workerOut.writeI32(index)

                val weightSizeBlob = rootIn.readBytes(SIZE3D_BYTES)
                workerOut.writeExact(weightSizeBlob)

                val configSize = rootIn.readI32()
                workerOut.writeI32(configSize)

                val sliceLength = rootIn.readI32()
                workerOut.writeI32(sliceLength)
                forwardRootToWorker(workerOut, sliceLength)

                // Forward pointer configuration (2 blocks of 12 bytes each)
                workerOut.writeExact(rootIn.readBytes(PTR_CFG_BYTES))
                workerOut.writeExact(rootIn.readBytes(PTR_CFG_BYTES))

                // Forward operation-specific config if present
                val config: ByteArray? = if (configSize > 0) {
                    rootIn.readBytes(configSize).also { workerOut.writeExact(it) }
                } else {
                    null
                }

                allOps.add(
                    OpInfo(
                        code,
                        weightSizeBlob.size3dNBytes(),
                        weightSizeBlob.size3dY(),
                        weightSizeBlob.size3dX(),
                        configSize,
                        config
                    )
                )
            }
        }
        workerConn.flush()

        // Handshake round 4: worker acknowledges all config
        workerIn.readAck()
        rootOut.writeAck()
        rootFlush()

        // Calculate weight bytes and FLOP estimates for performance logging
        var weightBytes = 0L
        var matmulOpsPerToken = 0L
        for (op in allOps) {
            weightBytes += op.weightSizeNBytes
            // For matmul operations with config: extract expert information for MoE scaling
            if (op.code == OP_MATMUL && op.config != null && op.configSize >= 8) {
                val configBuffer = ByteBuffer.wrap(op.config).order(ByteOrder.LITTLE_ENDIAN)
                val nExperts = configBuffer.int
                val nActive = configBuffer.int
                val activeFactor = if (nExperts > 0) nActive.toLong() else 1L
                matmulOpsPerToken += activeFactor * op.weightY * op.weightX
            }
        }

        WorkerInfo(nodeIndex, nSegments, nBuffers, allOps, allSegments, weightBytes, matmulOpsPerToken)
    }
}

// ============================================================================
// WEIGHT FORWARDING: ROOT TO WORKERS (ROUND-ROBIN)
// ============================================================================

/**
 * Forwards model weights from the root node to all workers in a round-robin fashion.
 *
 * PROTOCOL:
 * For each weight tensor:
 * - Read tensor metadata: name, operation index, offset, size
 * - Forward to next worker in round-robin order
 * - When nameSize=0, current worker is done (send final ACK)
 *
 * Round-robin distribution evenly balances weight forwarding bandwidth across workers.
 *
 * @param rootIn Input stream from root node
 * @param rootOut Output stream to root node
 * @param rootFlush Lambda to flush root output stream
 * @param workers List of connected worker socket connections
 * @param workerAddrs Original (host, port) addresses for logging
 */
private fun forwardWeights(
    rootIn: InputStream,
    rootOut: OutputStream,
    rootFlush: () -> Unit,
    workers: List<SocketConn>,
    workerAddrs: List<Pair<String, Int>>
) {
    val nWorkers = workers.size
    val workerDone = BooleanArray(nWorkers)
    val workerTotalBytes = LongArray(nWorkers)
    val workerPacketCount = IntArray(nWorkers)
    var nWorkersFinished = 0
    var currentWorker = 0
    val weightChunkBuf = ByteArray(1024 * 1024)

    println("[Stage 3] Forwarding weights from root to $nWorkers worker(s)...")

    while (nWorkersFinished < nWorkers) {
        // Find next worker not yet finished
        while (workerDone[currentWorker]) currentWorker = (currentWorker + 1) % nWorkers

        val nameSize = rootIn.readI32()
        val workerConn = workers[currentWorker]

        if (nameSize == 0) {
            // End-of-weights marker for this worker
            workerConn.out.writeI32(nameSize)
            workerConn.flush()

            // Wait for worker acknowledgment
            val finalAck = workerConn.inp.readI32()
            rootOut.writeI32(finalAck)
            rootFlush()

            // Log final stats for this worker
            println(
                "   Worker[$currentWorker] (${workerAddrs[currentWorker].first}:${workerAddrs[currentWorker].second}): " +
                "${workerPacketCount[currentWorker]} packets, " +
                "${"%.2f".format(workerTotalBytes[currentWorker] / (1024.0 * 1024.0))} MB"
            )
            workerDone[currentWorker] = true
            nWorkersFinished++
        } else {
            // Forward weight tensor to current worker
            workerConn.out.writeI32(nameSize)

            val nameBytes = rootIn.readBytes(nameSize)
            workerConn.out.writeExact(nameBytes)
            val operationName = String(nameBytes, 0, nameSize - 1)

            val opIndex = rootIn.readI32()
            workerConn.out.writeI32(opIndex)

            val offset = rootIn.readI64()
            workerConn.out.writeI64(offset)

            val nBytes = rootIn.readI64()
            workerConn.out.writeI64(nBytes)

            // Forward weight data in chunks
            var remainingBytes = nBytes
            while (remainingBytes > 0) {
                val chunkSize = minOf(remainingBytes, weightChunkBuf.size.toLong()).toInt()
                rootIn.readExact(weightChunkBuf, 0, chunkSize)
                workerConn.out.writeExact(weightChunkBuf, 0, chunkSize)
                remainingBytes -= chunkSize
            }
            workerConn.flush()

            // Update statistics
            workerTotalBytes[currentWorker] += nBytes
            workerPacketCount[currentWorker]++
            println(
                "   Worker[$currentWorker] op=%-22s idx=%3d offset=%8d size=%8d kB".format(
                    operationName,
                    opIndex,
                    offset,
                    nBytes / 1024
                )
            )
        }

        currentWorker = (currentWorker + 1) % nWorkers
    }
    println("[Stage 3] All weights forwarded\n")
}

// ============================================================================
// INFERENCE RELAY LOOP
// ============================================================================

/**
 * Main inference loop: coordinates computation between root and workers.
 *
 * FLOW FOR EACH INFERENCE STEP:
 * 1. Read ControlPacket (position, batchSize) from root
 * 2. Forward ControlPacket to all workers
 * 3. For each synchronization point, orchestrate data exchange:
 *    - SYNC_WITH_ROOT: Root sends to all workers
 *    - SYNC_NODE_SLICES: Workers + root exchange slices (all-reduce style)
 *    - SYNC_NODE_SLICES_EXCEPT_ROOT: Only workers exchange slices
 * 4. Repeat until batchSize=0 (end signal)
 *
 * BATCHING:
 * Each step processes [batchSize] tokens. Synchronization is repeated for each token
 * to ensure correctness in pipeline-parallel computation.
 *
 * @param rootIn Input stream from root node
 * @param rootOut Output stream to root node
 * @param rootFlush Lambda to flush root output stream
 * @param workers List of connected worker socket connections
 * @param syncList Flattened list of synchronization points from all segments
 * @param pipeSizes Pipe size metadata for calculating token slice sizes
 * @param nNodes Total number of nodes (workers + root)
 * @param workerInfos Metadata about each worker (used for performance logging)
 * @param workerAddrs Original (host, port) addresses for logging
 */
private fun inferenceRelayLoop(
    rootIn: InputStream,
    rootOut: OutputStream,
    rootFlush: () -> Unit,
    workers: List<SocketConn>,
    syncList: List<SyncInfo>,
    pipeSizes: List<ByteArray>,
    nNodes: Int,
    workerInfos: List<WorkerInfo>,
    workerAddrs: List<Pair<String, Int>>
) {
    val nWorkers = workers.size
    var stepIndex = 0
    var syncDataBuf = ByteArray(0)

    println("[Stage 4/5] Entering inference relay loop...\n")

    while (true) {
        val controlPacket = rootIn.readControlPacket()

        if (controlPacket.batchSize == 0) {
            // End-of-inference signal
            println("Stop signal received, forwarding to all workers")
            for (workerConn in workers) {
                workerConn.out.writeControlPacket(controlPacket)
                workerConn.flush()
            }
            break
        }

        // Log step with performance estimate
        println("Step $stepIndex: position=${controlPacket.position} batchSize=${controlPacket.batchSize}")
        for (workerIdx in 0 until nWorkers) {
            val gflopsPerToken = workerInfos[workerIdx].matmulOpsPerToken * 2.0 / 1e9
            val gflopsForBatch = gflopsPerToken * controlPacket.batchSize
            println(
                "   Worker[$workerIdx] (${workerAddrs[workerIdx].first}:${workerAddrs[workerIdx].second}): " +
                "${"%.3f".format(gflopsForBatch)} GFLOPs"
            )
        }

        // Send control packet to all workers
        for (workerConn in workers) {
            workerConn.out.writeControlPacket(controlPacket)
            workerConn.flush()
        }

        val batchSize = controlPacket.batchSize

        // Process synchronization points for this step
        for (syncInfo in syncList) {
            val tokenBytes = pipeTokenBytes(pipeSizes[syncInfo.pipeIndex])
            val sliceTokenBytes = tokenBytes / nNodes

            // Process each token in the batch
            for (batchIdx in 0 until batchSize) {
                when (syncInfo.syncType) {

                    SYNC_WITH_ROOT -> {
                        // Root sends data to all workers
                        if (syncDataBuf.size < tokenBytes) syncDataBuf = ByteArray(tokenBytes)
                        rootIn.readExact(syncDataBuf, 0, tokenBytes)
                        for (workerConn in workers) workerConn.out.writeExact(syncDataBuf, 0, tokenBytes)
                        for (workerConn in workers) workerConn.flush()
                    }

                    SYNC_NODE_SLICES -> {
                        // All-gather: every node contributes its slice and receives all others.
                        // In isolated/proxy mode each worker has nSockets=1 so the worker:
                        //   1. Sends its own slice (buffer[nodeIndex * sliceBytes]) to the proxy.
                        //   2. Reads nNodes-1 slices back, in ascending slice-index order
                        //      skipping its own, placing each at buffer[sliceIndex * sliceBytes].
                        //
                        // Proxy protocol:
                        //   root   → proxy : root's slice[0]              (sliceTokenBytes)
                        //   worker → proxy : worker's slice[nodeIndex]    (sliceTokenBytes each)
                        //   proxy  → root  : all worker slices[1..nNodes-1]
                        //   proxy  → worker w : all slices except slice[w+1], in order
                        val totalSliceBytes = sliceTokenBytes * nNodes
                        if (syncDataBuf.size < totalSliceBytes) syncDataBuf = ByteArray(totalSliceBytes)

                        // Gather: root contributes slice[0]
                        rootIn.readExact(syncDataBuf, 0, sliceTokenBytes)

                        // Gather: each worker contributes its slice
                        for (workerIdx in 0 until nWorkers) {
                            workers[workerIdx].inp.readExact(syncDataBuf, (workerIdx + 1) * sliceTokenBytes, sliceTokenBytes)
                        }

                        // Send all worker slices to root
                        for (nodeIdx in 1 until nNodes) {
                            rootOut.writeExact(syncDataBuf, nodeIdx * sliceTokenBytes, sliceTokenBytes)
                        }
                        rootFlush()

                        // Send all other slices to each worker sequentially, in ascending
                        // slice-index order skipping the worker's own slice.
                        // Worker w (nodeIndex = w+1) reads nNodes-1 slices from the proxy,
                        // placing each at buffer[sliceIndex * sliceBytes] for sliceIndex != w+1.
                        for (workerIdx in 0 until nWorkers) {
                            val workerNodeIndex = workerIdx + 1
                            for (sliceIndex in 0 until nNodes) {
                                if (sliceIndex == workerNodeIndex) continue // worker already has its own slice
                                workers[workerIdx].out.writeExact(syncDataBuf, sliceIndex * sliceTokenBytes, sliceTokenBytes)
                            }
                            workers[workerIdx].flush()
                        }
                    }

                    SYNC_NODE_SLICES_EXCEPT_ROOT -> {
                        // Only workers participate: workers exchange slices without root
                        val totalWorkerSliceBytes = sliceTokenBytes * nWorkers
                        if (syncDataBuf.size < totalWorkerSliceBytes) syncDataBuf = ByteArray(totalWorkerSliceBytes)

                        // Gather: each worker contributes its slice
                        for (workerIdx in 0 until nWorkers) {
                            workers[workerIdx].inp.readExact(syncDataBuf, workerIdx * sliceTokenBytes, sliceTokenBytes)
                        }

                        // Distribute: root receives all worker slices
                        for (workerIdx in 0 until nWorkers) {
                            rootOut.writeExact(syncDataBuf, workerIdx * sliceTokenBytes, sliceTokenBytes)
                        }
                        rootFlush()
                    }
                }
            }
        }

        println("   Step $stepIndex done")
        stepIndex++
    }

    println("Inference session complete after $stepIndex steps")
}

// ============================================================================
// COMMAND-LINE ARGUMENT PARSING
// ============================================================================

/**
 * Contains parsed command-line arguments for the proxy.
 *
 * @param workerAddrs List of (host, port) pairs for worker connections
 * @param bindHost Host address to listen on (default: 0.0.0.0)
 * @param bindPort Port to listen on (default: 9990)
 */
private data class ProxyArgs(
    val workerAddrs: List<Pair<String, Int>>,
    val bindHost: String,
    val bindPort: Int
)

/**
 * Parses command-line arguments for the proxy server.
 *
 * USAGE:
 *   kotlin DllamaProxy.kt --workers <host:port> [<host:port>...] [--host <bind>] [--port <port>]
 *
 * EXAMPLE:
 *   kotlin DllamaProxy.kt --workers worker1:9001 worker2:9001 --host 0.0.0.0 --port 9990
 *
 * @param args Command-line arguments
 * @return Parsed ProxyArgs
 * @throws IllegalArgumentException if required arguments are missing or malformed
 */
private fun parseArgs(args: Array<String>): ProxyArgs {
    var bindHost = "0.0.0.0"
    var bindPort = 9990
    val workers = mutableListOf<Pair<String, Int>>()
    var i = 0

    while (i < args.size) {
        when (args[i]) {
            "--workers" -> {
                i++
                // Collect all worker addresses until next flag
                while (i < args.size && !args[i].startsWith("-")) {
                    val parts = args[i].split(":")
                    require(parts.size == 2) { "Worker address must be host:port, got: ${args[i]}" }
                    workers.add(parts[0] to parts[1].toInt())
                    i++
                }
                continue
            }
            "--host" -> {
                bindHost = args[++i]
            }
            "--port" -> {
                bindPort = args[++i].toInt()
            }
            else -> error("Unknown option: ${args[i]}")
        }
        i++
    }

    require(workers.isNotEmpty()) { "Proxy requires at least one worker (--workers host:port ...)" }
    return ProxyArgs(workers, bindHost, bindPort)
}

// ============================================================================
// MAIN: PROXY SERVER
// ============================================================================

/**
 * Main entry point for the dllama proxy server.
 *
 * FIVE-STAGE PROCESS:
 * [Stage 1] Connect to all workers in isolated mode
 * [Stage 2] Forward model configuration from root to workers
 * [Stage 3] Forward model weights from root to workers (round-robin)
 * [Stage 4] Enter inference relay loop for each batch
 * [Stage 5] Complete inference session
 *
 * The proxy runs an infinite loop, accepting multiple root connections sequentially.
 * Each connection goes through all 5 stages before the server awaits the next root node.
 *
 * @param args Command-line arguments (--workers, --host, --port)
 */
fun main(args: Array<String>) {
    val proxyConfig = parseArgs(args)
    val nWorkers = proxyConfig.workerAddrs.size
    val nNodes = nWorkers + 1  // workers + root

    println("Proxy starting on ${proxyConfig.bindHost}:${proxyConfig.bindPort}, forwarding to $nWorkers worker(s)")

    println("\n[Stage 1] Connecting to $nWorkers worker(s) in isolated mode...")
    val workerConns = connectWorkersIsolated(proxyConfig.workerAddrs)
    println("[Stage 1] All $nWorkers worker(s) connected\n")

    // Main server loop: accept root connections sequentially
    while (true) {
        val serverSocket = ServerSocket()
        serverSocket.reuseAddress = true
        serverSocket.bind(InetSocketAddress(proxyConfig.bindHost, proxyConfig.bindPort))
        println("Waiting for root node to connect on ${proxyConfig.bindHost}:${proxyConfig.bindPort}...")
        val rootSocket = serverSocket.accept()
        serverSocket.close()
        rootSocket.tcpNoDelay = true
        println("Root node connected")

        val rootConn = rootSocket.buffered()
        val rootFlush = { rootConn.flush() }

        try {
            // Handshake: send number of nodes and wait for ACK
            rootConn.out.writeI32(nNodes)
            rootConn.flush()
            rootConn.inp.readAck()
            println("Handshake complete: nNodes=$nNodes\n")

            // Stage 2: Forward model configuration
            println("[Stage 2] Forwarding config from root to $nWorkers worker(s)...")
            val pipeSizes = mutableListOf<ByteArray>()
            val workerInfos = forwardConfig(rootConn.inp, rootConn.out, rootFlush, workerConns, pipeSizes)

            println("\n[Stage 2] Computation per worker:")
            for (workerIdx in 0 until nWorkers) {
                println(
                    "   Worker[$workerIdx] (${proxyConfig.workerAddrs[workerIdx].first}:${proxyConfig.workerAddrs[workerIdx].second}): " +
                    "weights=${"%.2f".format(workerInfos[workerIdx].weightBytes / (1024.0 * 1024.0))} MB, " +
                    "matmul=${"%.3f".format(workerInfos[workerIdx].matmulOpsPerToken * 2.0 / 1e9)} GFLOPs/token"
                )
            }
            println()

            // Stage 3: Forward model weights
            forwardWeights(rootConn.inp, rootConn.out, rootFlush, workerConns, proxyConfig.workerAddrs)

            // Stage 4/5: Inference relay loop
            val syncList = workerInfos.firstOrNull()?.segments?.flatten() ?: emptyList()
            inferenceRelayLoop(
                rootConn.inp,
                rootConn.out,
                rootFlush,
                workerConns,
                syncList,
                pipeSizes,
                nNodes,
                workerInfos,
                proxyConfig.workerAddrs
            )

        } catch (e: Exception) {
            println("Network error: ${e.message}")
        } finally {
            rootConn.close()
        }
    }
}
