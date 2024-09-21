#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../Constants.h"
#include "../SECPK1/SECP256k1.h"

#define KSIZE 10
#define ITEM_SIZE   56
#define ITEM_SIZE32 (ITEM_SIZE/4)

class GPUEngine {
public:
  GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound);
  ~GPUEngine();
  void SetParams(uint64_t dpMask,Int *distance,Int *px,Int *py);
  void SetKangaroos(Int *px,Int *py,Int *d);
  void GetKangaroos(Int *px,Int *py,Int *d);
  void SetKangaroo(uint64_t kIdx,Int *px,Int *py,Int *d);
  bool Launch(std::vector<ITEM> &hashFound,bool spinWait = false);
  void SetWildOffset(Int *offset);
  int GetNbThread();
  int GetGroupSize();
  int GetMemory();
  bool callKernelAndWait();
  bool callKernel();

  std::string deviceName;

  static void *AllocatePinnedMemory(size_t size);
  static void FreePinnedMemory(void *buff);
  static void PrintCudaInfo();
  static bool GetGridSize(int gpuId,int *x,int *y);

private:
  Int wildOffset;
  int nbThread;
  int nbThreadPerGroup;
  uint64_t *inputKangaroo;
  uint64_t *inputKangarooPinned;
  uint32_t *outputItem;
  uint32_t *outputItemPinned;
  uint64_t *jumpPinned;
  bool initialised;
  bool lostWarning;
  uint32_t maxFound;
  uint32_t outputSize;
  uint32_t kangarooSize;
  uint32_t kangarooSizePinned;
  uint32_t jumpSize;
  uint64_t dpMask;
};

#endif // GPUENGINEH
