#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
using namespace std;

uint32_t Kangaroo::CheckHash(uint32_t h, uint32_t nbItem, HashTable* hT, FILE* f) {
    bool ok = true;
    vector<Int> dists;
    vector<uint32_t> types;
    Point Z;
    Z.Clear();
    uint32_t nbWrong = 0;

    unique_ptr<ENTRY[]> items;
    if (!hT) {
        items = make_unique<ENTRY[]>(nbItem);
        fread(items.get(), sizeof(ENTRY), nbItem, f);
    }

    for (uint32_t i = 0; i < nbItem; i++) {
        const ENTRY* e = hT ? hT->E[h].items[i] : &items[i];
        Int dist;
        uint32_t kType;
        HashTable::CalcDistAndType(e->d, &dist, &kType);
        dists.push_back(dist);
        types.push_back(kType);
    }

    vector<Point> P = secp->ComputePublicKeys(dists);
    vector<Point> Sp;
    for (uint32_t i = 0; i < nbItem; i++) {
        if (types[i] == TAME) {
            Sp.push_back(Z);
        } else {
            Sp.push_back(keyToSearch);
        }
    }

    vector<Point> S = secp->AddDirect(Sp, P);
    for (uint32_t i = 0; i < nbItem; i++) {
        const ENTRY* e = hT ? hT->E[h].items[i] : &items[i];
        uint32_t hC = S[i].x.bits64[2] & HASH_MASK;
        ok = (hC == h) && (S[i].x.bits64[0] == e->x.i64[0]) && (S[i].x.bits64[1] == e->x.i64[1])  && (S[i].x.bits64[2] == e->x.i64[2])  && (S[i].x.bits64[3] == e->x.i64[3]);
        if (!ok) nbWrong++;
    }

    return nbWrong;
}

bool Kangaroo::CheckPartition(TH_PARAM* p) {
    uint32_t part = p->hStart;
    string pName = string(p->part1Name);
    ifstream f1(pName + "/header", ios::binary);
    if (!f1.is_open()) return false;

    uint32_t hStart = part * (HASH_SIZE / MERGE_PART);
    uint32_t hStop = (part + 1) * (HASH_SIZE / MERGE_PART);
    p->hStart = 0;

    for (uint32_t h = hStart; h < hStop; h++) {
        uint32_t nbItem, maxItem;
        f1.read(reinterpret_cast<char*>(&nbItem), sizeof(uint32_t));
        f1.read(reinterpret_cast<char*>(&maxItem), sizeof(uint32_t));
        if (nbItem == 0) continue;
        p->hStop += CheckHash(h, nbItem, nullptr, nullptr);
        p->hStart += nbItem;
    }

    return true;
}

bool Kangaroo::CheckWorkFile(TH_PARAM* p) {
    uint32_t nWrong = 0;
    for (uint32_t h = p->hStart; h < p->hStop; h++) {
        if (hashTable.E[h].nbItem == 0) continue;
        nWrong += CheckHash(h, hashTable.E[h].nbItem, &hashTable, nullptr);
    }
    p->hStop = nWrong;
    return true;
}

void* _checkPartThread(void* lpParam) {
    auto p = reinterpret_cast<TH_PARAM*>(lpParam);
    p->obj->CheckPartition(p);
    p->isRunning = false;
    return nullptr;
}

void* _checkWorkThread(void* lpParam) {
    auto p = reinterpret_cast<TH_PARAM*>(lpParam);
    p->obj->CheckWorkFile(p);
    p->isRunning = false;
    return nullptr;
}

void Kangaroo::CheckPartition(int nbCore,std::string& partName) {

  double t0;
  double t1;
  uint32_t v1;

  t0 = Timer::getTick();

  FILE* f1 = ReadHeader(partName+"/header",&v1,HEADW);
  if(f1 == NULL)
    return;

  uint32_t dp1;
  Point k1;
  uint64_t count1;
  double time1;
  Int RS1;
  Int RE1;

  // Read global param
  ::fread(&dp1,sizeof(uint32_t),1,f1);
  ::fread(&RS1.bits64,32,1,f1); RS1.bits64[4] = 0;
  ::fread(&RE1.bits64,32,1,f1); RE1.bits64[4] = 0;
  ::fread(&k1.x.bits64,32,1,f1); k1.x.bits64[4] = 0;
  ::fread(&k1.y.bits64,32,1,f1); k1.y.bits64[4] = 0;
  ::fread(&count1,sizeof(uint64_t),1,f1);
  ::fread(&time1,sizeof(double),1,f1);

  k1.z.SetInt32(1);
  if(!secp->EC(k1)) {
    ::printf("CheckPartition: key1 does not lie on elliptic curve\n");
    ::fclose(f1);
    return;
  }

  ::fclose(f1);

  // Set starting parameters
  keysToSearch.clear();
  keysToSearch.push_back(k1);
  keyIdx = 0;
  collisionInSameHerd = 0;
  rangeStart.Set(&RS1);
  rangeEnd.Set(&RE1);
  InitRange();
  InitSearchKey();

  int l2 = (int)log2(nbCore);
  int nbThread = (int)pow(2.0,l2);
  if(nbThread > MERGE_PART) nbThread = MERGE_PART;

  ::printf("Thread: %d\n",nbThread);
  ::printf("CheckingPart");

  TH_PARAM* params = (TH_PARAM*)malloc(nbThread * sizeof(TH_PARAM));
  THREAD_HANDLE* thHandles = (THREAD_HANDLE*)malloc(nbThread * sizeof(THREAD_HANDLE));
  memset(params,0,nbThread * sizeof(TH_PARAM));
  uint64_t nbDP = 0;
  uint64_t nbWrong = 0;

  for(int p = 0; p < MERGE_PART; p += nbThread) {

    printf(".");

    for(int i = 0; i < nbThread; i++) {
      params[i].threadId = i;
      params[i].isRunning = true;
      params[i].hStart = p + i;
      params[i].hStop = 0;
      params[i].part1Name = strdup(partName.c_str());
      thHandles[i] = LaunchThread(_checkPartThread,params + i);
    }

    JoinThreads(thHandles,nbThread);

    for(int i = 0; i < nbThread; i++) {
      free(params[i].part1Name);
      nbDP += params[i].hStart;
      nbWrong += params[i].hStop;
    }

  }

  free(params);
  free(thHandles);

  t1 = Timer::getTick();

  double O = (double)nbWrong / (double)nbDP;
  O = (1.0-O) * 100.0;

  ::printf("[%.3f%% OK][%s]\n",O,GetTimeStr(t1 - t0).c_str());
  if(nbWrong>0) {

    ::printf("DP: %" PRId64 "\n",nbDP);
    ::printf("DP Wrong: %" PRId64 "\n",nbWrong);

  }

}

void Kangaroo::CheckWorkFile(int nbCore,std::string& fileName) {

  double t0;
  double t1;
  uint32_t v1;

  setvbuf(stdout,NULL,_IONBF,0);

  if(IsDir(fileName)) {
    CheckPartition(nbCore,fileName);
    return;
  }
    
  t0 = Timer::getTick();

  FILE* f1 = ReadHeader(fileName,&v1,HEADW);
  if(f1 == NULL)
    return;

  uint32_t dp1;
  Point k1;
  uint64_t count1;
  double time1;
  Int RS1;
  Int RE1;

  // Read global param
  ::fread(&dp1,sizeof(uint32_t),1,f1);
  ::fread(&RS1.bits64,32,1,f1); RS1.bits64[4] = 0;
  ::fread(&RE1.bits64,32,1,f1); RE1.bits64[4] = 0;
  ::fread(&k1.x.bits64,32,1,f1); k1.x.bits64[4] = 0;
  ::fread(&k1.y.bits64,32,1,f1); k1.y.bits64[4] = 0;
  ::fread(&count1,sizeof(uint64_t),1,f1);
  ::fread(&time1,sizeof(double),1,f1);

  k1.z.SetInt32(1);
  if(!secp->EC(k1)) {
    ::printf("CheckWorkFile: key1 does not lie on elliptic curve\n");
    ::fclose(f1);
    return;
  }

  // Set starting parameters
  keysToSearch.clear();
  keysToSearch.push_back(k1);
  keyIdx = 0;
  collisionInSameHerd = 0;
  rangeStart.Set(&RS1);
  rangeEnd.Set(&RE1);
  InitRange();
  InitSearchKey();

  int l2 = (int)log2(nbCore);
  int nbThread = (int)pow(2.0,l2);
  uint64_t nbDP = 0;
  uint64_t nbWrong = 0;

  ::printf("Thread: %d\n",nbThread);
  ::printf("Checking");

  TH_PARAM* params = (TH_PARAM*)malloc(nbThread * sizeof(TH_PARAM));
  THREAD_HANDLE* thHandles = (THREAD_HANDLE*)malloc(nbThread * sizeof(THREAD_HANDLE));
  memset(params,0,nbThread * sizeof(TH_PARAM));

  int block = HASH_SIZE / 64;

  for(int s = 0; s < HASH_SIZE; s += block) {

    ::printf(".");

    uint32_t S = s;
    uint32_t E = s + block;

    // Load hashtables
    hashTable.LoadTable(f1,S,E);

    int stride = block / nbThread;

    for(int i = 0; i < nbThread; i++) {
      params[i].threadId = i;
      params[i].isRunning = true;
      params[i].hStart = S + i * stride;
      params[i].hStop = S + (i + 1) * stride;
      thHandles[i] = LaunchThread(_checkWorkThread,params + i);
    }
    JoinThreads(thHandles,nbThread);

    for(int i = 0; i < nbThread; i++)
      nbWrong += params[i].hStop;
    nbDP += hashTable.GetNbItem();

    hashTable.Reset();

  }

  ::fclose(f1);
  free(params);
  free(thHandles);

  t1 = Timer::getTick();

  double O = (double)nbWrong / (double)nbDP;
  O = (1.0 - O) * 100.0;

  ::printf("[%.3f%% OK][%s]\n",O,GetTimeStr(t1 - t0).c_str());
  if(nbWrong > 0) {
    ::printf("DP: %" PRId64 "\n",nbDP);
    ::printf("DP Wrong: %" PRId64 "\n",nbWrong);

  }

}


void Kangaroo::Check() {

  Int::Check();

  initDPSize = 8;
  SetDP(initDPSize);

  double t0;
  double t1;
  int nbKey = 16384;
  vector<Point> pts1;
  vector<Point> pts2;
  vector<Int> priv;

  // Check on ComputePublicKeys
  for(int i = 0; i<nbKey; i++) {
    Int rnd;
    rnd.Rand(256);
    priv.push_back(rnd);
  }

  t0 = Timer::getTick();
  for(int i = 0; i<nbKey; i++)
    pts1.push_back(secp->ComputePublicKey(&priv[i]));
  t1 = Timer::getTick();
  ::printf("ComputePublicKey %d : %.3f KKey/s\n",nbKey,(double)nbKey / ((t1 - t0)*1000.0));

  t0 = Timer::getTick();
  pts2 = secp->ComputePublicKeys(priv);
  t1 = Timer::getTick();
  ::printf("ComputePublicKeys %d : %.3f KKey/s\n",nbKey,(double)nbKey / ((t1 - t0)*1000.0));

  bool ok = true;
  int i = 0;
  for(; ok && i<nbKey;) {
    ok = pts1[i].equals(pts2[i]);
    if(ok) i++;
  }

  if(!ok) {
    ::printf("ComputePublicKeys wrong at %d\n",i);
    ::printf("%s\n",pts1[i].toString().c_str());
    ::printf("%s\n",pts2[i].toString().c_str());
  }

}
