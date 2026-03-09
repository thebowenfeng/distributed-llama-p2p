#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"

#define ROOT_SOCKET_INDEX 0

void initSockets();
void cleanupSockets();
int acceptSocket(int serverSocket);
void setReuseAddr(int socket);
void writeSocket(int socket, const void* data, NnSize size);
void readSocket(int socket, void* data, NnSize size);
int createServerSocket(const char *host, const int port);
void destroySocket(int serverSocket);

class NnConnectionSocketException : public std::runtime_error {
public:
    NnConnectionSocketException(const std::string message);
};

class NnTransferSocketException : public std::runtime_error {
public:
    int code;
    NnTransferSocketException(int code, const std::string message);
};

class NnSocket {
public:
    int fd;
    NnSocket();
    NnSocket(int fd);
    ~NnSocket();
    void assign(int fd);
    int release();
};

struct NnSocketIo {
    NnUint socketIndex;
    const void *data;
    NnSize size;
};

class NnNetwork {
private:
    int *sockets;
    NnSize *sentBytes;
    NnSize *recvBytes;

public:
    static std::unique_ptr<NnNetwork> serve(const char *host, const int port);
    static std::unique_ptr<NnNetwork> connect(NnUint nSockets, char **hosts, NnUint *ports);
    static std::unique_ptr<NnNetwork> connectIsolated(NnUint nSockets, char **hosts, NnUint *ports, NnUint totalNodes);

    NnUint nSockets;
    NnUint nNodes; // logical node count (may differ from nSockets in proxy mode)

    NnNetwork(std::vector<NnSocket> *sockets);
    ~NnNetwork();

    void setTurbo(bool enabled);
    int getSocket(NnUint socketIndex) const;
    void write(const NnUint socketIndex, const void *data, const NnSize size);
    void read(const NnUint socketIndex, void *data, const NnSize size);
    void writeAck(const NnUint socketIndex);
    void readAck(const NnUint socketIndex);
    bool tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts);
    void writeMany(NnUint n, NnSocketIo *ios);
    void writeAll(void *data, NnSize size);
    void readMany(NnUint n, NnSocketIo *ios);
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
    void resetStats();
};

// A root-side network that communicates exclusively with a proxy node.
// The proxy aggregates all worker sync traffic, so from the root's perspective
// there is exactly one socket (to the proxy), but the proxy handles fanning
// out to all real workers. The interface is the same as NnNetwork so the rest
// of the inference stack needs no changes.
class NnProxyNetwork {
private:
    int proxyFd;
    NnSize _sentBytes;
    NnSize _recvBytes;
public:
    NnUint nNodes; // total logical nodes (root + nWorkers), set by proxy handshake

    static std::unique_ptr<NnProxyNetwork> connect(const char *proxyHost, NnUint proxyPort);

    NnProxyNetwork(int proxyFd, NnUint nNodes);
    ~NnProxyNetwork();

    void setTurbo(bool enabled);
    void write(const void *data, NnSize size);
    void read(void *data, NnSize size);
    void writeAll(void *data, NnSize size); // broadcasts to proxy (which fans out to workers)
    bool tryReadWithMaxAttempts(void *data, NnSize size, unsigned long maxAttempts);
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
    void resetStats();
    int getSocket() const;
};

// Synchronizer that uses NnProxyNetwork for root↔proxy communication.
class NnProxyNodeSynchronizer : public NnNodeSynchronizer {
private:
    NnProxyNetwork *network;
    NnNetExecution *execution;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
public:
    NnProxyNodeSynchronizer(NnProxyNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    ~NnProxyNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

// Config writer that sends config to proxy (which forwards to each worker).
class NnProxyConfigWriter {
private:
    NnProxyNetwork *network;
public:
    NnProxyConfigWriter(NnProxyNetwork *network);
    void writeNet(NnNetConfig *config);
    void writeNode(NnNodeConfig *config);
    void writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs);
};

// Weight loader that sends weight slices to proxy (which forwards to workers).
class NnProxyWeightLoader {
private:
    NnExecutor *executor;
    NnProxyNetwork *network;
    NnUint nNodes;
    NnByte *temp;
    NnSize tempSize;
    void allocate(NnSize size);
    void writeWeight(const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
public:
    NnProxyWeightLoader(NnExecutor *executor, NnProxyNetwork *network, NnUint nNodes);
    ~NnProxyWeightLoader();
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, NnUint opIndex, NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, NnUint opIndex, NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight);
    void finish();
};

class NnNetworkNodeSynchronizer : public NnNodeSynchronizer {
private:
    NnNetwork *network;
    NnNetExecution *execution;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
public:
    NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    ~NnNetworkNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

class NnRootConfigWriter {
private:
    NnNetwork *network;
public:
    NnRootConfigWriter(NnNetwork *network);
    void writeNet(NnUint socketIndex, NnNetConfig *config);
    void writeNode(NnUint socketIndex, NnNodeConfig *config);
    void writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs);
};

class NnWorkerConfigReader {
private:
    NnNetwork *network;
public:
    NnWorkerConfigReader(NnNetwork *network);
    NnNetConfig readNet();
    NnNodeConfig readNode();
};

class NnRootWeightLoader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnUint nNodes;
    NnByte *temp;
    NnSize tempSize;
public:
    NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes);
    ~NnRootWeightLoader();
    void writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight);
    void finish();
private:
    void allocate(NnSize size);};

class NnWorkerWeightReader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnByte *temp;
    NnUint tempSize;
public:
    NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network);
    ~NnWorkerWeightReader();
    void read();
private:
    void allocate(NnUint size);
};

#endif
