#include "app.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#ifndef _WIN32
#include <sys/select.h>
#include <sys/socket.h>
#else
#include <winsock2.h>
#endif
#if defined(DLLAMA_VULKAN)
    #include "nn/nn-vulkan.hpp"
#endif

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.host = "0.0.0.0";
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    args.proxyHost = nullptr;
    args.proxyPort = 0;

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    // First see if any of the args are asking for help/usage and fail fast
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(separator + 1);
            }

            i += count - 1;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--host") == 0) {
            args.host = value;
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else if (std::strcmp(name, "--proxy") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == nullptr)
                throw std::runtime_error("Invalid proxy address, expected host:port");
            int hostLen = separator - value;
            char *proxyHost = new char[hostLen + 1];
            std::memcpy(proxyHost, value, hostLen);
            proxyHost[hostLen] = '\0';
            args.proxyHost = proxyHost;
            args.proxyPort = std::atoi(separator + 1);
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    std::vector<NnExecutorDevice> devices;

    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("This build does not support GPU");
#endif
    }

    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0)) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution), -1, -1));
    }
    return devices;
}

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network, NnProxyNetwork *proxyNetwork, std::vector<NnSize> nodeMatmulOpsPerToken) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network;
    this->proxyNetwork = proxyNetwork;
    this->nodeMatmulOpsPerToken = nodeMatmulOpsPerToken;
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
    controlPacket.batchSize = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    controlPacket.position = position;
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
}

void RootLlmInference::forward() {
    bool hasRemote = (network != nullptr || proxyNetwork != nullptr);
    if (hasRemote) {
        printf("🔀 Distributing inference: position=%u batchSize=%u\n",
            controlPacket.position, controlPacket.batchSize);
        for (NnUint i = 0; i < (NnUint)nodeMatmulOpsPerToken.size(); i++)
            printf("   Node %u: %.3f GFLOPs\n", i, nodeMatmulOpsPerToken[i] * 2.0 * controlPacket.batchSize / 1e9);
        if (network != nullptr)
            network->writeAll(&controlPacket, sizeof(LlmControlPacket));
        else
            proxyNetwork->writeAll(&controlPacket, sizeof(LlmControlPacket));
    }
    executor->forward();
}

void RootLlmInference::finish() {
    controlPacket.batchSize = 0;
    if (network != nullptr)
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    else if (proxyNetwork != nullptr)
        proxyNetwork->writeAll(&controlPacket, sizeof(LlmControlPacket));
}

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->positionPipe = (float *)execution->pipes[0];
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &controlPacket, sizeof(LlmControlPacket), maxAttempts))
        return false;
    if (controlPacket.batchSize == 0) {
        printf("🛑 Stop signal\n");
        isFinished = true;
        return true;
    }
    for (NnUint i = 0; i < controlPacket.batchSize; i++)
        positionPipe[i] = (float)(controlPacket.position + i);
    execution->setBatchSize(controlPacket.batchSize);
    return true;
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
    if (header.weightType == F_Q40 && header.syncType != F_Q80)
        throw std::runtime_error("This version supports only Q40 weights with Q80 sync type");

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);

    // Connect to proxy first if specified, so we can get the real nNodes before building LlmNet
    std::unique_ptr<NnProxyNetwork> proxyNetworkPtr(nullptr);
    NnProxyNetwork *proxyNetwork = nullptr;
    if (args->proxyHost != nullptr) {
        proxyNetworkPtr = NnProxyNetwork::connect(args->proxyHost, args->proxyPort);
        proxyNetwork = proxyNetworkPtr.get();
    }

    NnUint nNodes = (proxyNetwork != nullptr) ? proxyNetwork->nNodes : args->nWorkers + 1;

    if (nNodes > header.nKvHeads)
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");

    LlmNet net = buildLlmNet(&header, nNodes, args->nBatches);
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    if (args->info) {
        tokenizer.printHeader();
        printLlmHeader(&header);
        printNodeRequiredMemory(&net.netConfig, rootNodeConfig);
    }

    NnNetExecution execution(args->nThreads, &net.netConfig);

    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    // Compute per-node matmul ops for logging (used in both proxy and direct modes)
    auto computeNodeMatmulOps = [&]() {
        std::vector<NnSize> ops(nNodes, 0);
        printf("📋 Computation per node (%u workers):\n", args->nWorkers);
        for (NnUint i = 0; i < nNodes; i++) {
            NnNodeConfig *nodeConfig = &net.nodeConfigs[i];
            NnSize weightBytes = 0;
            for (NnUint s = 0; s < nodeConfig->nSegments; s++) {
                NnSegmentConfig *segment = &nodeConfig->segments[s];
                for (NnUint o = 0; o < segment->nOps; o++) {
                    NnOpConfig *op = &segment->ops[o];
                    weightBytes += op->weightSize.nBytes;
                    if (op->code == OP_MATMUL) {
                        NnMatmulOpConfig *matmulConfig = (NnMatmulOpConfig *)op->config;
                        NnUint activeFactor = (matmulConfig->nExperts > 0) ? matmulConfig->nActiveExperts : 1;
                        ops[i] += (NnSize)activeFactor * (NnSize)op->weightSize.y * (NnSize)op->weightSize.x;
                    }
                }
            }
            if (i == 0) {
                printf("   Node 0 (root): weights=%.2f MB, matmul=%.3f GFLOPs/token\n",
                    weightBytes / (1024.0f * 1024.0f), ops[i] * 2.0 / 1e9);
            } else if (args->workerHosts != nullptr) {
                printf("   Node %u (%s:%u): weights=%.2f MB, matmul=%.3f GFLOPs/token\n",
                    i, args->workerHosts[i - 1], args->workerPorts[i - 1],
                    weightBytes / (1024.0f * 1024.0f), ops[i] * 2.0 / 1e9);
            } else {
                printf("   Node %u (via proxy): weights=%.2f MB, matmul=%.3f GFLOPs/token\n",
                    i, weightBytes / (1024.0f * 1024.0f), ops[i] * 2.0 / 1e9);
            }
        }
        return ops;
    };

    std::vector<NnSize> nodeMatmulOpsPerToken;

    if (proxyNetwork != nullptr) {
        synchronizer.reset(new NnProxyNodeSynchronizer(proxyNetwork, &execution, &net.netConfig, rootNodeConfig));
        NnProxyConfigWriter configWriter(proxyNetwork);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
        nodeMatmulOpsPerToken = computeNodeMatmulOps();
    } else if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        // Direct mode: root connects to workers directly
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig));
        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
        nodeMatmulOpsPerToken = computeNodeMatmulOps();
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution);
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    if (proxyNetwork != nullptr) {
        NnProxyWeightLoader weightLoader(&executor, proxyNetwork, nNodes);
        loadLlmNetWeight(args->modelPath, &net, &weightLoader);
    } else {
        NnRootWeightLoader weightLoader(&executor, network, nNodes);
        loadLlmNetWeight(args->modelPath, &net, &weightLoader);
    }

    RootLlmInference inference(&net, &execution, &executor, network, proxyNetwork, nodeMatmulOpsPerToken);

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }
    }
    if (proxyNetwork != nullptr && args->netTurbo) {
        proxyNetwork->setTurbo(true);
        printf("🚁 Network is in non-blocking mode\n");
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.network = network;
    context.executor = &executor;

    handler(&context);

    inference.finish();
}

void runProxyApp(AppCliArgs *args) {
    if (args->nWorkers == 0)
        throw std::runtime_error("Proxy requires at least one worker (--workers)");

    printf("🔀 Proxy starting on %s:%u, forwarding to %u worker(s)\n",
        args->host, args->port, args->nWorkers);

    NnUint nNodes = args->nWorkers + 1; // root + workers

    // -------------------------------------------------------------------------
    // Stage 1: Connect to all workers first, then accept root connection
    // -------------------------------------------------------------------------
    printf("\n📡 [Stage 1] Connecting to %u worker(s)...\n", args->nWorkers);
    std::unique_ptr<NnNetwork> workerNetwork(NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts).release());
    printf("✅ [Stage 1] All %u worker(s) connected\n\n", args->nWorkers);

    printf("✅ [Stage 1] Handshake complete: proxy <-> %u worker(s)\n\n", args->nWorkers);

    // Keep accepting root connections indefinitely (like worker mode)
    while (true) {
        printf("📡 Waiting for root node to connect on %s:%u...\n", args->host, args->port);
        int serverFd2 = createServerSocket(args->host, args->port);
        int rootFd = acceptSocket(serverFd2);
        destroySocket(serverFd2);
        printf("✅ Root node connected\n");

        // Send nNodes to root
        writeSocket(rootFd, &nNodes, sizeof(nNodes));
        { NnUint ack; readSocket(rootFd, &ack, sizeof(ack)); }
        printf("✅ Handshake complete with root, nNodes=%u\n\n", nNodes);

    // -------------------------------------------------------------------------
    // Stage 2: Forward config (NnNetConfig + NnNodeConfig) root → each worker
    // -------------------------------------------------------------------------
    printf("📋 [Stage 2] Forwarding config from root to %u worker(s)...\n", args->nWorkers);

    // Helper lambdas
    std::vector<NnByte> buf;
    auto r2w = [&](NnUint w, NnSize n) {
        if (buf.size() < n) buf.resize(n);
        readSocket(rootFd, buf.data(), n);
        workerNetwork->write(w, buf.data(), n);
    };
    auto w2r = [&](NnUint w, NnSize n) {
        if (buf.size() < n) buf.resize(n);
        workerNetwork->read(w, buf.data(), n);
        writeSocket(rootFd, buf.data(), n);
    };
    auto r2wAll = [&](NnSize n) {
        if (buf.size() < n) buf.resize(n);
        readSocket(rootFd, buf.data(), n);
        workerNetwork->writeAll(buf.data(), n);
    };
    auto ackR2W = [&](NnUint w) {
        NnUint a; readSocket(rootFd, &a, sizeof(a));
        workerNetwork->write(w, &a, sizeof(a));
    };
    auto ackW2R = [&](NnUint w) {
        NnUint a; workerNetwork->read(w, &a, sizeof(a));
        writeSocket(rootFd, &a, sizeof(a));
    };

    struct ProxyOpInfo {
        NnUint code;
        NnSize3D weightSize;
        NnUint configSize;
        NnByte *config;
    };
    struct ProxySyncInfo {
        NnUint pipeIndex;
        NnUint syncType;
    };
    struct ProxySegmentInfo {
        std::vector<ProxySyncInfo> syncs;
    };
    struct ProxyWorkerInfo {
        NnUint nodeIndex;
        NnUint nSegments;
        NnUint nBuffers;
        std::vector<ProxyOpInfo> ops;
        std::vector<ProxySegmentInfo> segments;
        NnSize weightBytes;
        NnSize matmulOpsPerToken;
    };
    // Pipe sizes (same for all workers, set from first worker's net config)
    std::vector<NnSize3D> pipeSizes;
    NnUint netNNodes = 0;
    std::vector<ProxyWorkerInfo> workerInfos(args->nWorkers);

    for (NnUint w = 0; w < args->nWorkers; w++) {
        printf("📋 [Stage 2] Forwarding config for worker[%u]...\n", w);
        // writeNet: ack → fields → worker ack back
        ackR2W(w);
        NnUint nb; readSocket(rootFd, &nb, sizeof(nb)); workerNetwork->write(w, &nb, sizeof(nb));
        NnUint nn; readSocket(rootFd, &nn, sizeof(nn)); workerNetwork->write(w, &nn, sizeof(nn));
        if (w == 0) netNNodes = nn;
        NnUint np; readSocket(rootFd, &np, sizeof(np)); workerNetwork->write(w, &np, sizeof(np));
        for (NnUint p = 0; p < np; p++) {
            NnSize3D sz; readSocket(rootFd, &sz, sizeof(sz)); workerNetwork->write(w, &sz, sizeof(sz));
            if (w == 0) pipeSizes.push_back(sz);
            NnUint sl; readSocket(rootFd, &sl, sizeof(sl)); workerNetwork->write(w, &sl, sizeof(sl));
            r2w(w, sl);
        }
        NnUint nps; readSocket(rootFd, &nps, sizeof(nps)); workerNetwork->write(w, &nps, sizeof(nps));
        for (NnUint ps = 0; ps < nps; ps++) {
            NnUint pi; readSocket(rootFd, &pi, sizeof(pi)); workerNetwork->write(w, &pi, sizeof(pi));
        }
        ackW2R(w);
        // writeNode: ack → fields → worker ack back
        ackR2W(w);
        NnUint ni; readSocket(rootFd, &ni, sizeof(ni)); workerNetwork->write(w, &ni, sizeof(ni));
        workerInfos[w].nodeIndex = ni;
        NnUint nBufs; readSocket(rootFd, &nBufs, sizeof(nBufs)); workerNetwork->write(w, &nBufs, sizeof(nBufs));
        workerInfos[w].nBuffers = nBufs;
        NnUint nSegs; readSocket(rootFd, &nSegs, sizeof(nSegs)); workerNetwork->write(w, &nSegs, sizeof(nSegs));
        workerInfos[w].nSegments = nSegs;
        for (NnUint b = 0; b < nBufs; b++) {
            NnSize3D sz; readSocket(rootFd, &sz, sizeof(sz)); workerNetwork->write(w, &sz, sizeof(sz));
            NnUint sl; readSocket(rootFd, &sl, sizeof(sl)); workerNetwork->write(w, &sl, sizeof(sl));
            r2w(w, sl);
        }
        for (NnUint s = 0; s < nSegs; s++) {
            ProxySegmentInfo segInfo;
            NnUint nSyncs; readSocket(rootFd, &nSyncs, sizeof(nSyncs)); workerNetwork->write(w, &nSyncs, sizeof(nSyncs));
            NnUint nOps;   readSocket(rootFd, &nOps,   sizeof(nOps));   workerNetwork->write(w, &nOps,   sizeof(nOps));
            for (NnUint sy = 0; sy < nSyncs; sy++) {
                NnUint pi; readSocket(rootFd, &pi, sizeof(pi)); workerNetwork->write(w, &pi, sizeof(pi));
                NnUint st; readSocket(rootFd, &st, sizeof(st)); workerNetwork->write(w, &st, sizeof(st));
                if (w == 0) segInfo.syncs.push_back({pi, st});
            }
            for (NnUint o = 0; o < nOps; o++) {
                ProxyOpInfo opInfo = {0, {}, 0, nullptr};
                NnUint code; readSocket(rootFd, &code, sizeof(code)); workerNetwork->write(w, &code, sizeof(code));
                opInfo.code = code;
                NnUint idx;  readSocket(rootFd, &idx,  sizeof(idx));  workerNetwork->write(w, &idx,  sizeof(idx));
                NnSize3D wsz; readSocket(rootFd, &wsz, sizeof(wsz)); workerNetwork->write(w, &wsz, sizeof(wsz));
                opInfo.weightSize = wsz;
                NnUint cfgSz; readSocket(rootFd, &cfgSz, sizeof(cfgSz)); workerNetwork->write(w, &cfgSz, sizeof(cfgSz));
                opInfo.configSize = cfgSz;
                NnUint sl; readSocket(rootFd, &sl, sizeof(sl)); workerNetwork->write(w, &sl, sizeof(sl));
                r2w(w, sl);
                NnPointerConfig inp, outp;
                readSocket(rootFd, &inp, sizeof(inp)); workerNetwork->write(w, &inp, sizeof(inp));
                readSocket(rootFd, &outp, sizeof(outp)); workerNetwork->write(w, &outp, sizeof(outp));
                if (cfgSz > 0) {
                    opInfo.config = new NnByte[cfgSz];
                    readSocket(rootFd, opInfo.config, cfgSz);
                    workerNetwork->write(w, opInfo.config, cfgSz);
                }
                workerInfos[w].ops.push_back(opInfo);
            }
            if (w == 0) workerInfos[w].segments.push_back(segInfo);
        }
        ackW2R(w);
        printf("✅ [Stage 2] Config forwarded for worker[%u] (nodeIndex=%u, nSegments=%u)\n",
            w, workerInfos[w].nodeIndex, workerInfos[w].nSegments);
    }

    // Compute per-worker weight bytes and GFLOPs from parsed op info
    printf("\n📋 [Stage 2] Computation per worker (from forwarded config):\n");
    for (NnUint w = 0; w < args->nWorkers; w++) {
        NnSize weightBytes = 0;
        NnSize matmulOps = 0;
        for (auto &op : workerInfos[w].ops) {
            weightBytes += op.weightSize.nBytes;
            if (op.code == OP_MATMUL && op.config != nullptr) {
                NnMatmulOpConfig *mc = (NnMatmulOpConfig *)op.config;
                NnUint af = (mc->nExperts > 0) ? mc->nActiveExperts : 1;
                matmulOps += (NnSize)af * (NnSize)op.weightSize.y * (NnSize)op.weightSize.x;
            }
        }
        workerInfos[w].weightBytes = weightBytes;
        workerInfos[w].matmulOpsPerToken = matmulOps;
        printf("   Worker[%u] (%s:%u): weights=%.2f MB, matmul=%.3f GFLOPs/token\n",
            w, args->workerHosts[w], args->workerPorts[w],
            weightBytes / (1024.0f * 1024.0f), matmulOps * 2.0 / 1e9);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // Stage 3: Forward weight packets root → each worker
    // -------------------------------------------------------------------------
    printf("💿 [Stage 3] Forwarding weights from root to %u worker(s)...\n", args->nWorkers);
    for (NnUint w = 0; w < args->nWorkers; w++) {
        NnSize totalFwd = 0;
        NnUint nPkts = 0;
        printf("💿 [Stage 3] Worker[%u] (%s:%u)...\n", w, args->workerHosts[w], args->workerPorts[w]);
        while (true) {
            NnUint nameSize;
            readSocket(rootFd, &nameSize, sizeof(nameSize));
            workerNetwork->write(w, &nameSize, sizeof(nameSize));
            if (nameSize == 0) { ackW2R(w); break; }
            r2w(w, nameSize);
            std::string opName((char *)buf.data(), nameSize - 1);
            NnUint opIdx; readSocket(rootFd, &opIdx, sizeof(opIdx)); workerNetwork->write(w, &opIdx, sizeof(opIdx));
            NnSize offset; readSocket(rootFd, &offset, sizeof(offset)); workerNetwork->write(w, &offset, sizeof(offset));
            NnSize nBytes; readSocket(rootFd, &nBytes, sizeof(nBytes)); workerNetwork->write(w, &nBytes, sizeof(nBytes));
            r2w(w, nBytes);
            totalFwd += nBytes; nPkts++;
            printf("💿 [Stage 3]   Worker[%u] op=%-22s idx=%3u offset=%8zu size=%8zu kB\n",
                w, opName.c_str(), opIdx, (size_t)offset, nBytes / 1024);
        }
        printf("✅ [Stage 3] Worker[%u]: %u packets, %.2f MB forwarded (expected %.2f MB)\n",
            w, nPkts, totalFwd / (1024.0f * 1024.0f),
            workerInfos[w].weightBytes / (1024.0f * 1024.0f));
    }
    printf("✅ [Stage 3] All weights forwarded\n\n");

    // -------------------------------------------------------------------------
    // Stage 4 & 5: Relay control packets and sync traffic (inference loop)
    // -------------------------------------------------------------------------
    // Protocol per forward call (matching NnProxyNodeSynchronizer on root side
    // and NnNetworkNodeSynchronizer on worker side):
    //
    //   root → proxy: LlmControlPacket
    //   proxy → workers: LlmControlPacket (writeAll)
    //
    //   Per segment, per sync:
    //     SYNC_WITH_ROOT (pipeBytes = full pipe):
    //       root → proxy: pipeBytes * batchSize
    //       proxy → all workers: broadcast same bytes
    //
    //     SYNC_NODE_SLICES (sliceBytes = pipeBytes / nNodes):
    //       root → proxy: sliceBytes * batchSize  (root's slice)
    //       each worker → proxy: sliceBytes * batchSize  (worker's slice)
    //       proxy → root: all worker slices (nWorkers * sliceBytes * batchSize)
    //       proxy → each worker: all OTHER nodes' slices ((nNodes-1) * sliceBytes * batchSize)
    //
    //     SYNC_NODE_SLICES_EXCEPT_ROOT (sliceBytes = pipeBytes / nNodes):
    //       each worker → proxy: sliceBytes * batchSize
    //       proxy → root: all worker slices (nWorkers * sliceBytes * batchSize)

    printf("🔀 [Stage 4/5] Entering inference relay loop...\n\n");

    // Build flat sync list from workerInfos[0] segments (same layout for all workers)
    // Each entry: {pipeIndex, syncType}
    std::vector<ProxySyncInfo> syncList;
    if (!workerInfos.empty()) {
        for (auto &seg : workerInfos[0].segments)
            for (auto &s : seg.syncs)
                syncList.push_back(s);
    }

    // Helper: get bytes per batch item for a pipe (full pipe size / batchSize is NOT right;
    // pipeSize.x = dim per token, pipeSize.y = nBatches; we want bytes for one full pipe pass)
    // pipeSizes[i].x = columns, pipeSizes[i].y = rows (nBatches), pipeSizes[i].floatType
    // Total pipe bytes for batchSize tokens = batchSize * getBytes(floatType, x)
    auto getPipeSliceBytes = [&](NnUint pipeIndex, NnUint batchSize) -> NnSize {
        NnSize3D &ps = pipeSizes[pipeIndex];
        return (NnSize)batchSize * getBytes(ps.floatType, ps.x);
    };

    // Scratch buffer for sync relay
    std::vector<NnByte> sliceBuf;

    LlmControlPacket controlPacket;
    NnUint stepIndex = 0;

    while (true) {
        readSocket(rootFd, &controlPacket, sizeof(LlmControlPacket));

        if (controlPacket.batchSize == 0) {
            printf("🛑 [Stage 4] Stop signal received, forwarding to all workers\n");
            workerNetwork->writeAll(&controlPacket, sizeof(LlmControlPacket));
            break;
        }

        printf("🔀 [Stage 4] Step %u: position=%u batchSize=%u\n",
            stepIndex, controlPacket.position, controlPacket.batchSize);
        for (NnUint w = 0; w < args->nWorkers; w++)
            printf("   Worker[%u] (%s:%u): %.3f GFLOPs\n",
                w, args->workerHosts[w], args->workerPorts[w],
                workerInfos[w].matmulOpsPerToken * 2.0 * controlPacket.batchSize / 1e9);

        workerNetwork->writeAll(&controlPacket, sizeof(LlmControlPacket));

        NnUint batchSize = controlPacket.batchSize;
        NnSize rootToWorkers = 0, workersToRoot = 0;

        // Handle each sync in the forward pass explicitly
        for (auto &sync : syncList) {
            NnSize pipeBytes = getPipeSliceBytes(sync.pipeIndex, batchSize);
            NnSize sliceBytes = pipeBytes / netNNodes;

            if (sync.syncType == SYNC_WITH_ROOT) {
                // root sends full pipe → proxy broadcasts to all workers
                if (sliceBuf.size() < pipeBytes) sliceBuf.resize(pipeBytes);
                readSocket(rootFd, sliceBuf.data(), pipeBytes);
                workerNetwork->writeAll(sliceBuf.data(), pipeBytes);
                rootToWorkers += pipeBytes;

            } else if (sync.syncType == SYNC_NODE_SLICES) {
                // Collect root's slice
                if (sliceBuf.size() < pipeBytes * netNNodes) sliceBuf.resize(pipeBytes * netNNodes);
                // root slice is at index 0
                readSocket(rootFd, sliceBuf.data(), sliceBytes);
                rootToWorkers += sliceBytes;

                // Collect each worker's slice
                for (NnUint w = 0; w < args->nWorkers; w++) {
                    NnByte *workerSlice = sliceBuf.data() + (w + 1) * sliceBytes;
                    workerNetwork->read(w, workerSlice, sliceBytes);
                    workersToRoot += sliceBytes;
                }

                // Send all worker slices to root
                for (NnUint w = 0; w < args->nWorkers; w++) {
                    NnByte *workerSlice = sliceBuf.data() + (w + 1) * sliceBytes;
                    writeSocket(rootFd, workerSlice, sliceBytes);
                }

                // Send all nodes' slices to each worker (all except that worker's own)
                for (NnUint w = 0; w < args->nWorkers; w++) {
                    for (NnUint n = 0; n < netNNodes; n++) {
                        if (n == w + 1) continue; // skip own slice
                        NnByte *nodeSlice = sliceBuf.data() + n * sliceBytes;
                        workerNetwork->write(w, nodeSlice, sliceBytes);
                    }
                }

            } else if (sync.syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                // Collect each worker's slice
                if (sliceBuf.size() < sliceBytes * args->nWorkers) sliceBuf.resize(sliceBytes * args->nWorkers);
                for (NnUint w = 0; w < args->nWorkers; w++) {
                    workerNetwork->read(w, sliceBuf.data() + w * sliceBytes, sliceBytes);
                    workersToRoot += sliceBytes;
                }
                // Send all worker slices to root
                for (NnUint w = 0; w < args->nWorkers; w++) {
                    writeSocket(rootFd, sliceBuf.data() + w * sliceBytes, sliceBytes);
                }
            }
        }

        printf("   Step %u done: root→workers=%zu kB, workers→root=%zu kB\n\n",
            stepIndex, rootToWorkers / 1024, workersToRoot / 1024);
        stepIndex++;
    }

        printf("✅ [Stage 4/5] Inference session complete after %u steps\n\n", stepIndex);
        destroySocket(rootFd);
    } // end while(true) — wait for next root connection
}

void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->host, args->port);
        NnNetwork *network = networkPtr.get();

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution);
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig);
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        NnWorkerWeightReader weightReader(&executor, network);
        weightReader.read();

        WorkerLlmInference inference(&execution, network);
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }
                executor.forward();
                isFirstAttempt = true;
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Network error: %s\n", e.what());
                break;
            } catch (const NnExecutorException &e) {
                printf("🚨 Inference error: %s\n", e.what());
                break;
            }
        }
    }
}
