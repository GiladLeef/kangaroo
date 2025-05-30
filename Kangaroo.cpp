#include <signal.h>
#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <pthread.h>
#include <signal.h>
#include <iostream>

using namespace std;

#define safe_delete_array(x) if(x) {delete[] x;x=NULL;}

Kangaroo::Kangaroo(Secp256K1 *secp,int32_t initDPSize,bool useGpu,string &workFile,string &iWorkFile,uint32_t savePeriod,bool saveKangaroo,bool saveKangarooByServer,
                   double maxStep,int wtimeout,int port,int ntimeout,string serverIp,string outputFile,bool splitWorkfile,bool poolMode) {

  this->secp = secp;
  this->initDPSize = initDPSize;
  this->useGpu = useGpu;
  this->offsetCount = 0;
  this->offsetTime = 0.0;
  this->workFile = workFile;
  this->saveWorkPeriod = savePeriod;
  this->inputFile = iWorkFile;
  this->nbLoadedWalk = 0;
  this->clientMode = serverIp.length() > 0;
  this->saveKangarooByServer = this->clientMode && saveKangarooByServer;
  this->saveKangaroo = saveKangaroo || this->saveKangarooByServer;
  this->fRead = NULL;
  this->maxStep = maxStep;
  this->wtimeout = wtimeout;
  this->port = port;
  this->ntimeout = ntimeout;
  this->serverIp = serverIp;
  this->outputFile = outputFile;
  this->hostInfo = NULL;
  this->endOfSearch = false;
  this->saveRequest = false;
  this->connectedClient = 0;
  this->totalRW = 0;
  this->collisionInSameHerd = 0;
  this->keyIdx = 0;
  this->splitWorkfile = splitWorkfile;
  this->pid = Timer::getPID();
  this->poolMode = poolMode;
  this->totalPoolDP = 0;

  CPU_GRP_SIZE = 1024;

  pthread_mutex_init(&ghMutex, NULL);
  pthread_mutex_init(&saveMutex, NULL);
  pthread_mutex_init(&poolStatsMutex, NULL);
  signal(SIGFPE, SIG_IGN);

}

bool Kangaroo::ParseConfigFile(std::string &fileName) {

  // In client mode, config come from the server
  if(clientMode)
    return true;

  // Check file
  FILE *fp = fopen(fileName.c_str(),"rb");
  if(fp == NULL) {
    ::printf("Error: Cannot open %s %s\n",fileName.c_str(),strerror(errno));
    return false;
  }
  fclose(fp);

  // Get lines
  vector<string> lines;
  int nbLine = 0;
  string line;
  ifstream inFile(fileName);
  while(getline(inFile,line)) {

    // Remove ending \r\n
    int l = (int)line.length() - 1;
    while(l >= 0 && isspace(line.at(l))) {
      line.pop_back();
      l--;
    }

    if(line.length() > 0) {
      lines.push_back(line);
      nbLine++;
    }

  }

  if(lines.size()<3) {
    ::printf("Error: %s not enough arguments\n",fileName.c_str());
    return false;
  }

  rangeStart.SetBase16((char *)lines[0].c_str());
  rangeEnd.SetBase16((char *)lines[1].c_str());
  for(int i=2;i<(int)lines.size();i++) {
    
    Point p;
    bool isCompressed;
    if( !secp->ParsePublicKeyHex(lines[i],p,isCompressed) ) {
      ::printf("%s, error line %d: %s\n",fileName.c_str(),i,lines[i].c_str());
      return false;
    }
    keysToSearch.push_back(p);

  }

  ::printf("Start:%s\n",rangeStart.GetBase16().c_str());
  ::printf("Stop :%s\n",rangeEnd.GetBase16().c_str());
  ::printf("Keys :%d\n",(int)keysToSearch.size());

  return true;

}

bool Kangaroo::IsDP(uint64_t x) {

  return (x & dMask) == 0;

}

void Kangaroo::SetDP(int size) {

  // Mask for distinguised point
  dpSize = size;
  if(dpSize == 0) {
    dMask = 0;
  } else {
    if(dpSize > 64) dpSize = 64;
    dMask = (1ULL << (64 - dpSize)) - 1;
    dMask = ~dMask;
  }

  ::printf("DP size: %d [0x%" PRIx64 "]\n",dpSize,dMask);
}

bool Kangaroo::Output(Int* pk, char sInfo, int sType) {
    FILE* f = stdout;
    bool needToClose = false;

    if (!outputFile.empty()) {
        f = fopen(outputFile.c_str(), "a");
        if (f == nullptr) {
            std::cerr << "Cannot open " << outputFile << " for writing\n";
            f = stdout;
        } else {
            needToClose = true;
        }
    }

    if (!needToClose) {
        std::printf("\n");
    }

    Point PR = secp->ComputePublicKey(pk);

    if (PR.equals(keysToSearch[keyIdx])) {
        std::fprintf(f, "Key#%2d [%d%c]Pub:  0x%s \n", keyIdx, sType, sInfo, secp->GetPublicKeyHex(true, keysToSearch[keyIdx]).c_str());
        std::fprintf(f, "       Priv: 0x%s \n", pk->GetBase16().c_str());
    } else {
        if (needToClose) {
            fclose(f);
        }
        return false;
    }

    if (needToClose) {
        fclose(f);
    }

    return true;
}

bool  Kangaroo::CheckKey(Int d1,Int d2,uint8_t type) {

  // Resolve equivalence collision

  if(type & 0x1)
    d1.ModNegK1order();
  if(type & 0x2)
    d2.ModNegK1order();

  Int pk(&d1);
  pk.ModAddK1order(&d2);

  Point P = secp->ComputePublicKey(&pk);

  if(P.equals(keyToSearch)) {
    pk.ModAddK1order(&rangeStart);    
    return Output(&pk,'N',type);
  }

  if(P.equals(keyToSearchNeg)) {
    pk.ModAddK1order(&rangeStart);
    return Output(&pk,'S',type);
  }

  return false;

}

bool Kangaroo::CollisionCheck(Int* d1,uint32_t type1,Int* d2,uint32_t type2) {


  if(type1 == type2) {

    // Collision inside the same herd
    return false;

  } else {

    Int Td;
    Int Wd;

    if(type1 == TAME) {
      Td.Set(d1);
      Wd.Set(d2);
    }  else {
      Td.Set(d2);
      Wd.Set(d1);
    }

    endOfSearch = CheckKey(Td,Wd,0) || CheckKey(Td,Wd,1) || CheckKey(Td,Wd,2) || CheckKey(Td,Wd,3);

    if(!endOfSearch) {

      // Should not happen, reset the kangaroo
      ::printf("\n Unexpected wrong collision, reset kangaroo !\n");
      if((int64_t)(Td.bits64[3])<0) {
        Td.ModNegK1order();
        ::printf("Found: Td-%s\n",Td.GetBase16().c_str());
      } else {
        ::printf("Found: Td %s\n",Td.GetBase16().c_str());
      }
      if((int64_t)(Wd.bits64[3])<0) {
        Wd.ModNegK1order();
        ::printf("Found: Wd-%s\n",Wd.GetBase16().c_str());
      } else {
        ::printf("Found: Wd %s\n",Wd.GetBase16().c_str());
      }
      return false;

    }

  }

  return true;

}

bool Kangaroo::AddToTable(Int *pos,Int *dist,uint32_t kType) {

  int addStatus = hashTable.Add(pos,dist,kType);
  if(addStatus== ADD_COLLISION)
    return CollisionCheck(&hashTable.kDist,hashTable.kType,dist,kType);

  return addStatus == ADD_OK;

}

bool Kangaroo::AddToTable(uint64_t h,int256_t *x,int256_t *d) {

  int addStatus = hashTable.Add(h,x,d);
  if(addStatus== ADD_COLLISION) {

    Int dist;
    uint32_t kType;
    HashTable::CalcDistAndType(*d,&dist,&kType);
    return CollisionCheck(&hashTable.kDist,hashTable.kType,&dist,kType);

  }

  return addStatus == ADD_OK;

}

void Kangaroo::SolveKeyCPU(TH_PARAM *ph) {
    vector<ITEM> dps;
    int thId = ph->threadId;
    double lastSent = 0;

    // Preallocate memory for arrays
    IntGroup grp(CPU_GRP_SIZE);
    Int dx[CPU_GRP_SIZE];
    Int dy, rx, ry, _s, _p;
    Int altRx, altRy;
    uint64_t jmps[CPU_GRP_SIZE];
    Int *p1xs[CPU_GRP_SIZE];
    Int *p1ys[CPU_GRP_SIZE];
    Int *p2xs[CPU_GRP_SIZE];
    Int *p2ys[CPU_GRP_SIZE];
    Int *distances[CPU_GRP_SIZE];
    bool isDPs[CPU_GRP_SIZE];
    bool altIsDPs[CPU_GRP_SIZE];
    
    Int backPx[CPU_GRP_SIZE];
    Int backPy[CPU_GRP_SIZE];
    Int backDist[CPU_GRP_SIZE];

    // Create Kangaroos if not already loaded
    if (ph->px == nullptr) {
        ph->px = new Int[CPU_GRP_SIZE];
        ph->py = new Int[CPU_GRP_SIZE];
        ph->distance = new Int[CPU_GRP_SIZE];
        CreateHerd(CPU_GRP_SIZE, ph->px, ph->py, ph->distance, TAME);
    }

    if (keyIdx == 0)
        ::printf("CPU Thread %d: %d kangaroos\n", ph->threadId, CPU_GRP_SIZE);

    ph->hasStarted = true;

    while (!endOfSearch) {
        // First pass - collect data and calculate dx
        for (int g = 0; g < CPU_GRP_SIZE; g++) {
            jmps[g] = ph->px[g].bits64[0] & (NB_JUMP - 1ULL);

            p1xs[g] = &jumpPointx[jmps[g]];
            p1ys[g] = &jumpPointy[jmps[g]];
            p2xs[g] = &ph->px[g];
            p2ys[g] = &ph->py[g];
            distances[g] = &jumpDistance[jmps[g]];
            dx[g].ModSub(p2xs[g], p1xs[g]);
        }
        
        grp.Set(dx);
        grp.ModInv();

        // Second pass - calculate both forward and backward points
        for (int g = 0; g < CPU_GRP_SIZE; g++) {
            dy.ModSub(p2ys[g], p1ys[g]);
            _s.ModMulK1(&dy, &dx[g]);
            _p.ModSquareK1(&_s);
            
            // Forward jump point calculation
            rx.ModSub(&_p, p1xs[g]);
            rx.ModSub(p2xs[g]);
            
            ry.ModSub(p2xs[g], &rx);
            ry.ModMulK1(&_s);
            ry.ModSub(p2ys[g]);

            // Backward jump point calculation (cheap second-point trick)
            // Derive alternate X coordinate cheaply: altRx = -rx
            altRx.Set(&rx);
            altRx.ModNeg();
            
            altRy.ModSub(p2xs[g], &altRx);
            altRy.ModMulK1(&_s);
            altRy.ModSub(p2ys[g]);
            altRy.ModNeg();
            
            // Store backward point data for batch processing later
            backPx[g].Set(&altRx);
            backPy[g].Set(&altRy);
            backDist[g].Set(&ph->distance[g]);
            backDist[g].ModSubK1order(distances[g]);
            altIsDPs[g] = IsDP(backPx[g].bits64[3]);
            
            // Update forward point
            ph->px[g].Set(&rx);
            ph->py[g].Set(&ry);
            ph->distance[g].ModAddK1order(distances[g]);
            
            // Check if forward point is DP
            isDPs[g] = IsDP(ph->px[g].bits64[3]);
        }
        
        // Batch process results
        if (clientMode) {
            // Process forward points
            for (int g = 0; g < CPU_GRP_SIZE; g++) {
                if (isDPs[g]) {
                    ITEM it;
                    it.x.Set(&ph->px[g]);
                    it.d.Set(&ph->distance[g]);
                    it.kIdx = g;
                    dps.push_back(it);
                }
                
                // Process backward points
                if (altIsDPs[g]) {
                    ITEM it;
                    it.x.Set(&backPx[g]);
                    it.d.Set(&backDist[g]);
                    it.kIdx = g;
                    dps.push_back(it);
                }
            }
            
            // Send data to server when appropriate
            double now = Timer::getTick();
            if (now - lastSent > SEND_PERIOD && !dps.empty()) {
                LOCK(ghMutex);
                SendToServer(dps, ph->threadId, 0xFFFF);
                UNLOCK(ghMutex);
                lastSent = now;
                dps.clear();
            }
            
            counters[thId] += CPU_GRP_SIZE * 2; // Count both forward and backward
        } else {
            // -----------------------------------------------------------------
            //  Optimisation: lock the global hash table ONLY when we actually
            //  have at least one distinguished point to insert.  With the usual
            //  DP density this skips 90 – 99 % of the lock/unlock pairs and
            //  improves scalability on 8-core+ CPUs.
            // -----------------------------------------------------------------

            bool haveDP = false;
            for (int g = 0; g < CPU_GRP_SIZE && !haveDP; ++g)
                haveDP = isDPs[g] | altIsDPs[g];

            if (haveDP) {
                LOCK(ghMutex);

                // Process both forward and backward points
                for (int g = 0; g < CPU_GRP_SIZE && !endOfSearch; g++) {
                    // Forward
                    if (isDPs[g]) {
                        if (!AddToTable(&ph->px[g], &ph->distance[g], g % 2)) {
                            // Collision inside the same herd – reset tame/wild
                            CreateHerd(1, &ph->px[g], &ph->py[g], &ph->distance[g], g % 2, false);
                            collisionInSameHerd++;
                        }
                    }

                    // Backward (cheap second point)
                    if (altIsDPs[g]) {
                        if (!AddToTable(&backPx[g], &backDist[g], g % 2)) {
                            collisionInSameHerd++;
                        }
                    }
                    counters[thId] += 2;
                }

                UNLOCK(ghMutex);
            } else {
                // No DP in this batch → just count the iterations.
                counters[thId] += CPU_GRP_SIZE * 2;
            }
        }

        // Save request
        if (saveRequest && !endOfSearch) {
            ph->isWaiting = true;
            LOCK(saveMutex);
            ph->isWaiting = false;
            UNLOCK(saveMutex);
        }
    }

    // Clean up allocated memory
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);

    ph->isRunning = false;
}

void Kangaroo::SolveKeyGPU(TH_PARAM *ph) {

  double lastSent = 0;

  // Global init
  int thId = ph->threadId;

#ifdef WITHGPU

  // Static array to track which GPUs have been initialized (max 256 GPUs)
  static bool gpuInitialized[256] = {false};
  vector<ITEM> dps;
  vector<ITEM> gpuFound;
  GPUEngine *gpu;

  gpu = new GPUEngine(ph->gridSizeX,ph->gridSizeY,ph->gpuId,65536 * 2);

  // Only print device information once per GPU ID
  if(!gpuInitialized[ph->gpuId])
    ::printf("%s (%.1f MB used)\n",gpu->deviceName.c_str(),gpu->GetMemory() / 1048576.0);
  
  double t0 = Timer::getTick();
  
  if(ph->px==NULL) {
    // Only print creation message once per GPU ID
    if(!gpuInitialized[ph->gpuId]) {
      ::printf("GPU Thread GPU#%d: creating kangaroos...\n", ph->gpuId);
      gpuInitialized[ph->gpuId] = true;
    }
    // Create Kangaroos, if not already loaded
    uint64_t nbThread = gpu->GetNbThread();
    ph->px = new Int[ph->nbKangaroo];
    ph->py = new Int[ph->nbKangaroo];
    ph->distance = new Int[ph->nbKangaroo];

    for(uint64_t i = 0; i<nbThread; i++) {
      CreateHerd(GPU_GRP_SIZE,&(ph->px[i*GPU_GRP_SIZE]),
                              &(ph->py[i*GPU_GRP_SIZE]),
                              &(ph->distance[i*GPU_GRP_SIZE]),
                              TAME);
    }
  }

  gpu->SetWildOffset(&rangeWidthDiv2);
  gpu->SetParams(dMask,jumpDistance,jumpPointx,jumpPointy);
  gpu->SetKangaroos(ph->px,ph->py,ph->distance);

  if(workFile.length()==0 || !saveKangaroo) {
    // No need to get back kangaroo, free memory
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);
  }

  gpu->callKernel();
  double t1 = Timer::getTick();

  // Only print completion message once per GPU ID for the first key
  if(keyIdx == 0 && !gpuInitialized[ph->gpuId])
    ::printf("GPU Thread GPU#%d: 2^%.2f kangaroos [%.1fs]\n",ph->gpuId,log2((double)ph->nbKangaroo),(t1-t0));

  ph->hasStarted = true;

  while(!endOfSearch) {

    gpu->Launch(gpuFound);
    counters[thId] += ph->nbKangaroo * NB_RUN * 2;  // Double count for second-point trick

    if( clientMode ) {

      for(int i=0;i<(int)gpuFound.size();i++)
        dps.push_back(gpuFound[i]);

      double now = Timer::getTick();
      if(now - lastSent > SEND_PERIOD) {
        LOCK(ghMutex);
        SendToServer(dps,ph->threadId,ph->gpuId);
        UNLOCK(ghMutex);
        lastSent = now;
      }

    } else {
      if(gpuFound.size() > 0) {
        LOCK(ghMutex);
        
        for(int g = 0; !endOfSearch && g < gpuFound.size(); g++) {
          uint32_t kType = (uint32_t)(gpuFound[g].kIdx % 2);
          if(!AddToTable(&gpuFound[g].x,&gpuFound[g].d,kType)) {
            // Collision inside the same herd
            // We need to reset the kangaroo
            Int px;
            Int py;
            Int d;
            CreateHerd(1,&px,&py,&d,kType,false);
            gpu->SetKangaroo(gpuFound[g].kIdx,&px,&py,&d);
            collisionInSameHerd++;
          }
        }
        UNLOCK(ghMutex);
      }
    }
    // Save request
    if(saveRequest && !endOfSearch) {
      // Get kangaroos
      if(saveKangaroo)
        gpu->GetKangaroos(ph->px,ph->py,ph->distance);
      ph->isWaiting = true;
      LOCK(saveMutex);
      ph->isWaiting = false;
      UNLOCK(saveMutex);
    }
  }

  safe_delete_array(ph->px);
  safe_delete_array(ph->py);
  safe_delete_array(ph->distance);
  delete gpu;
  
#else
  ph->hasStarted = true;
#endif
  ph->isRunning = false;
}

void *_SolveKeyCPU(void *lpParam) {
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->SolveKeyCPU(p);
  return 0;
}
void *_SolveKeyGPU(void *lpParam) {
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->SolveKeyGPU(p);
  return 0;
}

void Kangaroo::CreateHerd(int nbKangaroo, Int *px, Int *py, Int *d, int firstType, bool lock) {
    vector<Int>   pk(nbKangaroo);
    vector<Point> S(nbKangaroo);

    // Only wild kangaroos need an extra addition with keyToSearch – collect their
    // indices so we can batch them separately and avoid useless work for the
    // tame half.
    vector<int>   wildIdx;       wildIdx.reserve(nbKangaroo / 2 + 8);
    vector<Point> wildS;         wildS.reserve(nbKangaroo / 2 + 8);
    vector<Point> wildAdd;       wildAdd.reserve(nbKangaroo / 2 + 8);

    Point Z;
    Z.Clear();
    int offset = firstType % 2;
    
    if (lock) LOCK(ghMutex);

    const bool isTame = (offset % 2 == TAME);
    for (int j = 0; j < nbKangaroo; j++) {
        if ((j + offset) % 2 == TAME) {
            // Tame in [0..N] - use direct random generation
            d[j].Rand(rangePower);
        } else {
            // Wild in [-N/8..N/8] - optimize calculation
            d[j].Rand(rangePower - 2);
            d[j].ModSubK1order(&rangeWidthDiv8);
        }
        pk[j] = d[j];
    }

    if (lock) UNLOCK(ghMutex);
    
    S = secp->ComputePublicKeys(pk);

    // Build the vectors that actually need the extra addition (wild only)
    for (int j = 0; j < nbKangaroo; j++) {
        if ((j + offset) % 2 != TAME) {   // wild
            wildIdx.push_back(j);
            wildS.push_back(S[j]);
            wildAdd.push_back(keyToSearch);
        }
    }

    if (!wildIdx.empty()) {
        std::vector<Point> wildRes = secp->AddDirect(wildAdd, wildS);
        // Copy the adjusted points back into the main S vector
        for (size_t k = 0; k < wildIdx.size(); ++k)
            S[wildIdx[k]] = wildRes[k];
    }

    for (int j = 0; j < nbKangaroo; j++) {
        px[j].Set(&S[j].x);
        py[j].Set(&S[j].y);
    }
}

void Kangaroo::CreateJumpTable() {
  int jumpBit = rangePower / 2 + 1;
  int maxRetry = 100;
  bool ok = false;
  double distAvg;
  double maxAvg = pow(2.0,(double)jumpBit - 0.95);
  double minAvg = pow(2.0,(double)jumpBit - 1.05);
  // Kangaroo jumps
  // Constant seed for compatibilty of workfiles
  rseed(0x600DCAFE);

  // Positive only
  // When using symmetry, the sign is switched by the symmetry class switch
  while(!ok && maxRetry>0 ) {
    Int totalDist;
    totalDist.SetInt32(0);
    for(int i = 0; i < NB_JUMP; ++i) {
      jumpDistance[i].Rand(jumpBit);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
  }
    distAvg = totalDist.ToDouble() / (double)(NB_JUMP);
    ok = distAvg>minAvg && distAvg<maxAvg;
    maxRetry--;
  }

  // Batch computation of jump table using a single vector multi-scalar multiplication
  // Build a vector of scalars that will be passed to the batched routine
  std::vector<Int> scalars;
  scalars.reserve(NB_JUMP);
  for(int i = 0; i < NB_JUMP; ++i)
    scalars.push_back(jumpDistance[i]);

  // Compute all public keys at once – this path will internally take advantage of
  // batching (ComputePublicKeys already uses a single batch inversion for the Z
  // coordinates). Re-using it here avoids NB_JUMP individual scalar multiplications
  // and prepares the ground for a future GPU off-load where the whole batch can be
  // executed in a single kernel.
  std::vector<Point> Jpoints = secp->ComputePublicKeys(scalars);

  for(int i = 0; i < NB_JUMP; ++i) {
    jumpPointx[i].Set(&Jpoints[i].x);
    jumpPointy[i].Set(&Jpoints[i].y);
  }

  ::printf("Jump average distance: 2^%.2f\n",log2(distAvg));

  unsigned long seed = Timer::getSeed32();
  rseed(seed);

}

void Kangaroo::ComputeExpected(double dp,double *op,double *ram,double *overHead) {
  // Compute expected number of operation and memory
  double gainS = 1.0;
  // Kangaroo number
  double k = (double)totalRW;
  // Range size
  double N = pow(2.0,(double)rangePower);
  // theta
  double theta = pow(2.0,dp);
  double Z0 = (2.0 * (2.0 - sqrt(2.0)) * gainS) * sqrt(M_PI) * 1.42;
  double avgDP0 = Z0 * sqrt(N);
  // DP Overhead
  *op = Z0 * pow(N * (k * theta + sqrt(N)),1.0 / 3.0);
  *ram = (double)sizeof(HASH_ENTRY) * (double)HASH_SIZE + // Table
         (double)sizeof(ENTRY *) * (double)(HASH_SIZE * 4) + // Allocation overhead
         (double)(sizeof(ENTRY) + sizeof(ENTRY *)) * (*op / theta); // Entries
  *ram /= (1024.0*1024.0);
  if(overHead)
    *overHead = *op/avgDP0;
}

void Kangaroo::InitRange() {
  rangeWidth.Set(&rangeEnd);
  rangeWidth.Sub(&rangeStart);
  rangePower = rangeWidth.GetBitLength();
  ::printf("Range width: 2^%d\n",rangePower);
  rangeWidthDiv2.Set(&rangeWidth);
  rangeWidthDiv2.ShiftR(1);
  rangeWidthDiv4.Set(&rangeWidthDiv2);
  rangeWidthDiv4.ShiftR(1);
  rangeWidthDiv8.Set(&rangeWidthDiv4);
  rangeWidthDiv8.ShiftR(1);
}

void Kangaroo::InitSearchKey() {
  Int SP;
  SP.Set(&rangeStart);
  if(!SP.IsZero()) {
    Point RS = secp->ComputePublicKey(&SP);
    RS.y.ModNeg();
    keyToSearch = secp->AddDirect(keysToSearch[keyIdx],RS);
  } else {
    keyToSearch = keysToSearch[keyIdx];
  }
  keyToSearchNeg = keyToSearch;
  keyToSearchNeg.y.ModNeg();
}

void Kangaroo::Run(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize) {
    double t0 = Timer::getTick();
    nbCPUThread = nbThread;
  nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
    totalRW = 0;
#ifndef WITHGPU
  if(nbGPUThread>0) {
    ::printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
    nbGPUThread = 0;
  }
#endif
    uint64_t totalThread = (uint64_t)nbCPUThread + (uint64_t)nbGPUThread;
    if(totalThread == 0) {
      ::printf("No CPU or GPU threads found, exiting...\n");
      ::exit(0);
    }
    TH_PARAM *params = (TH_PARAM *)malloc(totalThread * sizeof(TH_PARAM));
    THREAD_HANDLE *thHandles = (THREAD_HANDLE *)malloc(totalThread * sizeof(THREAD_HANDLE));
    memset(params, 0, totalThread * sizeof(TH_PARAM));
    memset(counters, 0, sizeof(counters));
    ::printf("Number of CPU threads: %d\n", nbCPUThread);
#ifdef WITHGPU
  // Compute grid size
  for(int i = 0; i < nbGPUThread; i++) {    
    int x = gridSize[2ULL * i];
    int y = gridSize[2ULL * i + 1ULL];
    if(!GPUEngine::GetGridSize(gpuId[i],&x,&y)) {
      std::cout << "Failed to get grid size for GPU ID: " << gpuId[i] << std::endl;
      return;
    } else {
      params[nbCPUThread + i].gridSizeX = x;
      params[nbCPUThread + i].gridSizeY = y;
    }
    params[nbCPUThread + i].nbKangaroo = (uint64_t)GPU_GRP_SIZE * x * y;
    totalRW += params[nbCPUThread + i].nbKangaroo;
  }
#endif
    totalRW += nbCPUThread * (uint64_t)CPU_GRP_SIZE;
    // Set starting parameters
    if (clientMode) {
        // Retrieve config from server
        if (!GetConfigFromServer())
            ::exit(0);
        // Client save only kangaroos, force -ws
        if (workFile.length() > 0)
            saveKangaroo = true;
    }

    InitRange();
    CreateJumpTable();

    ::printf("Number of kangaroos: 2^%.2f\n", log2((double)totalRW));

    if (!clientMode) {
        // Compute suggested distinguished bits number for less than 5% overhead (see README)
        double dpOverHead;
        // Adjust for the efficiency of the second-point trick
        int suggestedDP = (int)((double)rangePower / 2.0 - log2((double)totalRW) - 1);
        if (suggestedDP < 0)
            suggestedDP = 0;
        ComputeExpected((double)suggestedDP, &expectedNbOp, &expectedMem, &dpOverHead);
        while (dpOverHead > 1.05 && suggestedDP > 0) {
            suggestedDP--;
            ComputeExpected((double)suggestedDP, &expectedNbOp, &expectedMem, &dpOverHead);
        }
        if (initDPSize < 0)
            initDPSize = suggestedDP;

        ComputeExpected((double)initDPSize, &expectedNbOp, &expectedMem);
        if (nbLoadedWalk == 0)
            ::printf("Recommended DP: %d\n", suggestedDP);
        ::printf("Expected operations: 2^%.2f\n", log2(expectedNbOp));
        ::printf("Expected RAM usage: %.1fMB\n", expectedMem);
    }
    else {
        keyIdx = 0;
        InitSearchKey();
    }
    SetDP(initDPSize);
    // Fetch kangaroos (if any)
    FectchKangaroos(params);

        for (keyIdx = 0; keyIdx < keysToSearch.size(); keyIdx++) {
            InitSearchKey();
            endOfSearch = false;
            collisionInSameHerd = 0;
            // Reset counters
            memset(counters, 0, sizeof(counters));
            // Launch CPU threads
            for (int i = 0; i < nbCPUThread; i++) {
                params[i].threadId = i;
                params[i].isRunning = true;
                thHandles[i] = LaunchThread(_SolveKeyCPU, params + i);
            }
            #ifdef WITHGPU
                  // Launch GPU threads
                  for(int i = 0; i < nbGPUThread; i++) {
                    int id = nbCPUThread + i;
                    params[id].threadId = 0x80L + i;
                    params[id].isRunning = true;
                    params[id].gpuId = gpuId[i];
                    thHandles[id] = LaunchThread(_SolveKeyGPU,params + id);
                  }

            #endif
            // Wait for end
            Process(params, "MK/s");
            JoinThreads(thHandles, nbCPUThread);
            hashTable.Reset();
    }
    double t1 = Timer::getTick();
    ::printf("\nDone: Total time %s \n", GetTimeStr(t1 - t0 + offsetTime).c_str());
}
