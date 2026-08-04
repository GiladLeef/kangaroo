#ifndef KANGAROO_H
#define KANGAROO_H

// Single shared declaration header for the whole solver; every CPU translation
// unit includes only this file. The pieces that must stay headers:
//   - constants.h : macros are preprocessor-only, they cannot be shared any
//                   other way.
//   - int.h       : Int is used inside the CUDA closure (engine.cu compiles
//                   for both host and nvptx targets and cannot load module
//                   units), so its definition must be textually included.
//   - the standard library: consolidated here instead of repeated per unit.
// The GPU closure (engine.h, kernel.h, compute.h) is deliberately NOT included
// here: engine.cu must stay independent and self-contained.

#include "constants.h"
#include "int.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
class Timer {
public:
  static void Init();
  static double getTick();
  static void printResult(char *unit, int nbTry, double t0, double t1);
  static std::string getResult(char *unit, int nbTry, double t0, double t1);
  static int getCoreNumber();
  static std::string getSeed(int size);
  static void SleepMillis(uint32_t millis);
  static uint32_t getSeed32();
  static uint32_t getPID();
  static std::string getTS();

  static time_t tickStart;
};

// ---------------------------------------------------------------------------
// Point (Jacobian coordinates)
// ---------------------------------------------------------------------------
class Point {
public:
  Point();
  Point(Int *cx, Int *cy, Int *cz);
  Point(Int *cx, Int *cz);
  Point(const Point &p);
  ~Point();
  bool isZero();
  bool equals(const Point &p) const;
  void Set(const Point &p);
  void Set(Int *cx, Int *cy, Int *cz);
  void Clear();
  void Reduce();
  std::string toString();

  Int x;
  Int y;
  Int z;
};

// ---------------------------------------------------------------------------
// IntGroup (batch modular inversion)
// ---------------------------------------------------------------------------
class IntGroup {
public:
  IntGroup(int size);
  ~IntGroup();
  void Set(Int *pts);
  void ModInv();

private:
  Int *ints;
  std::vector<Int> subp;
  int size;
};

// ---------------------------------------------------------------------------
// Secp256K1
// ---------------------------------------------------------------------------
class Secp256K1 {
public:
  Secp256K1();
  ~Secp256K1();
  void  Init();
  Point ComputePublicKey(Int *privKey, bool reduce = true);
  std::vector<Point> ComputePublicKeys(std::vector<Int> &privKeys);
  Point NextKey(Point &key);
  bool  EC(Point &p);

  std::string GetPublicKeyHex(bool compressed, Point &p);
  bool ParsePublicKeyHex(const std::string& str, Point &p, bool &isCompressed);

  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  std::vector<Point> AddDirect(std::vector<Point> &p1, std::vector<Point> &p2);

  Point G;                 // Generator
  Int   order;             // Curve order

private:
  uint8_t GetByte(const std::string &str, int idx);

  Int GetY(Int x, bool isEven);
  Point GTable[256 * 32];  // Generator table
};

// ---------------------------------------------------------------------------
// Work file helpers
// ---------------------------------------------------------------------------
namespace workfile {

struct WorkFilePayload {
  uint32_t dpSize = 0;
  Int rangeStart;
  Int rangeEnd;
  Point key;
  uint64_t totalCount = 0;
  double totalTime = 0.0;
};

struct FileCloser {
  void operator()(FILE* f) const noexcept {
    if (f != nullptr) {
      ::fclose(f);
    }
  }
};

using FileHandle = std::unique_ptr<FILE, FileCloser>;

inline FileHandle openFile(const std::string& path, const char* mode) {
  return FileHandle(::fopen(path.c_str(), mode));
}

inline bool readExact(FILE* f, void* dst, size_t size, size_t count = 1) {
  return ::fread(dst, size, count, f) == count;
}

inline bool writeExact(FILE* f, const void* src, size_t size, size_t count = 1) {
  return ::fwrite(src, size, count, f) == count;
}

template <typename T>
inline bool readValue(FILE* f, T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "readValue requires a trivially copyable type");
  return readExact(f, &value, sizeof(T));
}

template <typename T>
inline bool readValue(std::istream& is, T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "readValue requires a trivially copyable type");
  return static_cast<bool>(is.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template <typename T>
inline bool readValue(FileHandle& f, T& value) {
  return readValue(f.get(), value);
}

template <typename T>
inline bool writeValue(FILE* f, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "writeValue requires a trivially copyable type");
  return writeExact(f, &value, sizeof(T));
}

template <typename T>
inline bool writeValue(std::ostream& os, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "writeValue requires a trivially copyable type");
  return static_cast<bool>(os.write(reinterpret_cast<const char*>(&value), sizeof(T)));
}

template <typename T>
inline bool writeValue(FileHandle& f, const T& value) {
  return writeValue(f.get(), value);
}

inline bool readInt256(FILE* f, Int& value) {
  if (!readExact(f, &value.bits64, 32)) {
    return false;
  }
  value.bits64[4] = 0;
  return true;
}

inline bool readInt256(FileHandle& f, Int& value) {
  return readInt256(f.get(), value);
}

inline bool writeInt256(FILE* f, const Int& value) {
  return writeExact(f, &value.bits64, 32);
}

inline bool writeInt256(FileHandle& f, const Int& value) {
  return writeInt256(f.get(), value);
}

inline bool readPointAffine(FILE* f, Point& value) {
  if (!readInt256(f, value.x) || !readInt256(f, value.y)) {
    return false;
  }
  value.z.SetInt32(1);
  return true;
}

inline bool readPointAffine(FileHandle& f, Point& value) {
  return readPointAffine(f.get(), value);
}

inline bool writePointAffine(FILE* f, const Point& value) {
  return writeInt256(f, value.x) && writeInt256(f, value.y);
}

inline bool writePointAffine(FileHandle& f, const Point& value) {
  return writePointAffine(f.get(), value);
}

inline bool readWorkPayload(FILE* f, WorkFilePayload& payload) {
  return readExact(f, &payload.dpSize, sizeof(uint32_t)) &&
         readInt256(f, payload.rangeStart) &&
         readInt256(f, payload.rangeEnd) &&
         readPointAffine(f, payload.key) &&
         readExact(f, &payload.totalCount, sizeof(uint64_t)) &&
         readExact(f, &payload.totalTime, sizeof(double));
}

inline bool readWorkPayload(FileHandle& f, WorkFilePayload& payload) {
  return readWorkPayload(f.get(), payload);
}

inline bool writeWorkPayload(FILE* f, const WorkFilePayload& payload) {
  return writeExact(f, &payload.dpSize, sizeof(uint32_t)) &&
         writeInt256(f, payload.rangeStart) &&
         writeInt256(f, payload.rangeEnd) &&
         writePointAffine(f, payload.key) &&
         writeExact(f, &payload.totalCount, sizeof(uint64_t)) &&
         writeExact(f, &payload.totalTime, sizeof(double));
}

inline bool writeWorkPayload(FileHandle& f, const WorkFilePayload& payload) {
  return writeWorkPayload(f.get(), payload);
}

}  // namespace workfile

// ---------------------------------------------------------------------------
// Hash table
// ---------------------------------------------------------------------------
union int256_s {
  uint8_t  i8[32];
  uint16_t i16[16];
  uint32_t i32[8];
  uint64_t i64[4];
};

typedef union int256_s int256_t;

typedef struct {
  int256_t x;  // Position of kangaroo (256bit LSB)
  int256_t d;  // Travelled distance (b255=sign b254=kangaroo type, b253..b0 distance)
} ENTRY;

struct HASH_ENTRY {
  uint32_t nbItem = 0;
  std::vector<ENTRY> items;
};

class HashTable {
public:
  HashTable();
  int Add(Int *x, Int *d, uint32_t type);
  int Add(uint64_t h, int256_t *x, int256_t *d);
  int Add(uint64_t h, const ENTRY &e);
  uint64_t GetNbItem();
  uint32_t GetBucketSize(uint32_t h) const;
  const ENTRY* GetEntry(uint32_t h, uint32_t i) const;
  void Reset();
  std::string GetSizeInfo();
  void PrintInfo();
  void SaveTable(FILE *f);
  void SaveTable(FILE* f, uint32_t from, uint32_t to, bool printPoint = true);
  void LoadTable(FILE *f);
  void LoadTable(FILE* f, uint32_t from, uint32_t to);
  void SeekNbItem(FILE* f, bool restorePos = false);
  void SeekNbItem(FILE* f, uint32_t from, uint32_t to);

  HASH_ENTRY E[HASH_SIZE];
  // Collision info
  Int      kDist;
  uint32_t kType;

  static void Convert(Int *x, Int *d, uint32_t type, uint64_t *h, int256_t *X, int256_t *D);
  static int MergeH(uint32_t h, FILE* f1, FILE* f2, FILE* fd, uint32_t *nbDP, uint32_t* duplicate,
                    Int* d1, uint32_t* k1, Int* d2, uint32_t* k2);
  static void CalcDistAndType(int256_t d, Int* kDist, uint32_t* kType);

private:
  static ENTRY CreateEntry(int256_t *x, int256_t *d);
  static int compare(const int256_t *i1, const int256_t *i2);
  std::string GetStr(int256_t *i);
};

// ---------------------------------------------------------------------------
// Kangaroo: types, network protocol structs, and the main solver class
// ---------------------------------------------------------------------------
typedef int SOCKET;

using THREAD_HANDLE = std::thread;

class Kangaroo;  // forward declaration (referenced by TH_PARAM below)

// Input thread parameters
typedef struct {
  Kangaroo *obj;
  int  threadId;
  bool isRunning;
  bool hasStarted;
  bool isWaiting;
  uint64_t nbKangaroo;
  int  gridSizeX;
  int  gridSizeY;
  int  gpuId;
  std::vector<Int> px;      // Kangaroo position
  std::vector<Int> py;      // Kangaroo position
  std::vector<Int> distance; // Travelled distance

  SOCKET clientSock;
  std::string clientInfo;

  uint32_t hStart;
  uint32_t hStop;
  std::string part1Name;
  std::string part2Name;

} TH_PARAM;

// DP transferred over the network
typedef struct {
  uint32_t kIdx;
  uint32_t h;
  int256_t x;
  int256_t d;
} DP;

typedef struct {
  uint32_t header;
  uint32_t nbDP;
  uint32_t threadId;
  uint32_t processId;
} DPHEADER;

// DP cache
typedef struct {
  std::vector<DP> dp;
} DP_CACHE;

// Filesystem namespace alias used across implementation units
namespace fs = std::filesystem;

// Maximum length of a network-transferred filename
constexpr uint32_t MAX_NETWORK_FILENAME = 256;

class Kangaroo {
public:
  Kangaroo(Secp256K1 *secp, int32_t initDPSize, bool useGpu, std::string &workFile,
           std::string &iWorkFile, uint32_t savePeriod, bool saveKangaroo,
           bool saveKangarooByServer, double maxStep, int wtimeout, int port,
           int ntimeout, std::string serverIp, std::string outputFile, bool splitWorkfile);
  void Run(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize);
  void RunServer();
  bool ParseConfigFile(std::string &fileName);
  bool LoadWork(std::string &fileName);
  void Check();
  void MergeDir(std::string& dirname, std::string& dest);
  bool MergeWork(std::string &file1, std::string &file2, std::string &dest, bool printStat = true);
  void WorkInfo(std::string &fileName);
  bool MergeWorkPart(std::string& file1, std::string& file2, bool printStat);
  bool MergeWorkPartPart(std::string& part1Name, std::string& part2Name);
  static void CreateEmptyPartWork(std::string& partName);
  void CheckWorkFile(int nbCore, std::string& fileName);
  void CheckPartition(int nbCore, std::string& partName);
  bool FillEmptyPartFromFile(std::string& partName, std::string& fileName, bool printStat);

  // Threaded procedures
  void SolveKeyCPU(TH_PARAM *p);
  void SolveKeyGPU(TH_PARAM *p);
  bool HandleRequest(TH_PARAM *p);
  bool MergePartition(TH_PARAM* p);
  bool CheckPartition(TH_PARAM* p);
  bool CheckWorkFile(TH_PARAM* p);
  void ProcessServer();

  void AddConnectedClient();
  void RemoveConnectedClient();
  void RemoveConnectedKangaroo(uint64_t nb);

private:
  bool IsDP(uint64_t x);
  void SetDP(int size);
  void CreateHerd(int nbKangaroo, Int *px, Int *py, Int *d, int firstType, bool lock = true);
  void CreateJumpTable();
  bool AddToTable(uint64_t h, int256_t *x, int256_t *d);
  bool AddToTable(Int *pos, Int *dist, uint32_t kType);
  bool SendToServer(std::vector<ITEM> &dp, uint32_t threadId, uint32_t gpuId);
  bool CheckKey(Int d1, Int d2, uint8_t type);
  bool CollisionCheck(Int* d1, uint32_t type1, Int* d2, uint32_t type2);
  void ComputeExpected(double dp, double *op, double *ram, double* overHead = NULL);
  void InitRange();
  void InitSearchKey();
  bool SetSearchContext(const Int& start, const Int& end, const Point& key, const char* label);
  std::string GetTimeStr(double s);
  bool Output(Int* pk, char sInfo, int sType);
  bool LoadWorkPayload(const std::string& fileName, const char* label, uint32_t* version,
                       workfile::FileHandle* fOut, workfile::WorkFilePayload& payload);

  // Backup stuff
  void SaveWork(std::string fileName, FILE *f, int type, uint64_t totalCount, double totalTime);
  void SaveWork(uint64_t totalCount, double totalTime, TH_PARAM *threads, int nbThread);
  void SaveServerWork();
  void FetchWalks(uint64_t nbWalk, Int *x, Int *y, Int *d);
  void FetchWalks(uint64_t nbWalk, std::vector<int256_t>& kangs, Int* x, Int* y, Int* d);
  void FectchKangaroos(TH_PARAM *threads);
  workfile::FileHandle ReadHeader(std::string fileName, uint32_t *version, int type);
  bool  SaveHeader(std::string fileName, FILE* f, int type, uint64_t totalCount, double totalTime);
  int FSeek(FILE *stream, uint64_t pos);
  uint64_t FTell(FILE *stream);
  int IsDir(std::string dirName);
  bool IsEmpty(std::string fileName);
  static std::string GetPartName(const std::string& partName, int i, bool tmpPart);
  static FILE* OpenPart(const std::string& partName, char* mode, int i, bool tmpPart = false);
  uint32_t CheckHash(uint32_t h, uint32_t nbItem, HashTable* hT, FILE* f);

  // Network stuff
  void AcceptConnections(SOCKET server_soc);
  int WaitFor(SOCKET sock, int timeout, int mode);
  int Write(SOCKET sock, char *buf, int bufsize, int timeout);
  int Read(SOCKET sock, char *buf, int bufsize, int timeout);
  bool GetConfigFromServer();
  bool ConnectToServer(SOCKET *retSock);
  void WaitForServer();
  int32_t GetServerStatus();
  bool SendKangaroosToServer(std::string& fileName, std::vector<int256_t>& kangs);
  bool GetKangaroosFromServer(std::string& fileName, std::vector<int256_t>& kangs);

  std::mutex  ghMutex;
  std::mutex  saveMutex;
  THREAD_HANDLE LaunchThread(void *(*func) (void *), TH_PARAM *p);

  void JoinThreads(THREAD_HANDLE *handles, int nbThread);
  void Process(TH_PARAM *params, std::string unit);

  uint64_t getCPUCount();
  uint64_t getGPUCount();
  bool isAlive(TH_PARAM *p);
  bool hasStarted(TH_PARAM *p);
  bool isWaiting(TH_PARAM *p);

  Secp256K1 *secp;
  int  nbGPUThread;
  HashTable hashTable;
  std::array<uint64_t, 256> counters;
  int  nbCPUThread;
  double startTime;

  // Range
  int rangePower;
  Int rangeStart;
  Int rangeEnd;
  Int rangeWidth;
  Int rangeWidthDiv2;
  Int rangeWidthDiv4;
  Int rangeWidthDiv8;

  uint64_t dMask;
  uint32_t dpSize;
  int32_t initDPSize;
  uint64_t collisionInSameHerd;
  std::vector<Point> keysToSearch;
  Point keyToSearch;
  Point keyToSearchNeg;
  uint32_t keyIdx;
  bool endOfSearch;
  bool useGpu;

  double expectedNbOp;
  double expectedMem;
  double maxStep;
  uint64_t totalRW;

  Int jumpDistance[NB_JUMP];
  Int jumpPointx[NB_JUMP];
  Int jumpPointy[NB_JUMP];

  int CPU_GRP_SIZE;

  // Backup stuff
  std::string outputFile;
  workfile::FileHandle fRead;
  uint64_t offsetCount;
  double offsetTime;
  int64_t nbLoadedWalk;
  std::string workFile;
  std::string inputFile;
  int  saveWorkPeriod;
  bool saveRequest;
  bool saveKangaroo;
  bool saveKangarooByServer;
  int wtimeout;
  int ntimeout;
  bool splitWorkfile;

  // Network stuff
  int port;
  std::string lastError;
  std::string serverIp;
  std::vector<char> hostInfo;
  int   hostAddrType;
  bool  clientMode;
  bool  isConnected;
  SOCKET serverConn;
  std::vector<DP_CACHE> recvDP;
  std::vector<DP_CACHE> localCache;
  std::string serverStatus;
  int connectedClient;
  uint32_t pid;
};

#endif // KANGAROO_H