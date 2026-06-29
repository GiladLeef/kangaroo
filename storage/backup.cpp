#include "kangaroo.h"
#include <fstream>
#include "intgroup.h"
#include "timer.h"
#include "workfile.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <pthread.h>
#include <sys/stat.h>
#include <filesystem>

using namespace std;
using namespace workfile;
namespace fs = std::filesystem;

int Kangaroo::FSeek(FILE* stream,uint64_t pos) {
  return fseeko(stream,pos,SEEK_SET);
}

uint64_t Kangaroo::FTell(FILE* stream) {
  return (uint64_t)ftello(stream);
}

bool Kangaroo::IsEmpty(std::string fileName) {
  std::error_code ec;
  uint64_t size = fs::file_size(fileName, ec);
  if(ec) {
    ::printf("OpenPart: Cannot open %s for reading\n",fileName.c_str());
    ::printf("%s\n",ec.message().c_str());
    ::exit(0);
  }
  return size == 0;
}

int Kangaroo::IsDir(string dirName) {
  std::error_code ec;
  if(!fs::exists(dirName, ec)) {
    ::printf("%s not found\n",dirName.c_str());
    return -1;
  }
  return fs::is_directory(dirName, ec) ? 1 : 0;
}

FILE *Kangaroo::ReadHeader(std::string fileName, uint32_t *version, int type) {
  FILE *f = fopen(fileName.c_str(),"rb");
  if(f == NULL) {
    ::printf("ReadHeader: Cannot open %s for reading\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    return NULL;
  }
  uint32_t head;
  uint32_t versionF;

  // Read header
  if(::fread(&head,sizeof(uint32_t),1,f) != 1) {
    ::printf("ReadHeader: Cannot read from %s\n",fileName.c_str());
    if(::feof(f)) {
      ::printf("Empty file\n");
    } else {
      ::printf("%s\n",::strerror(errno));
    }
    ::fclose(f);
    return NULL;
  }

  ::fread(&versionF,sizeof(uint32_t),1,f);
  if(version) *version = versionF;

  if(head!=type) {
    if(head==HEADK) {
      readExact(f,&nbLoadedWalk,sizeof(uint64_t));
      ::printf("ReadHeader: %s is a kangaroo only file [2^%.2f kangaroos]\n",fileName.c_str(),log2((double)nbLoadedWalk));
    } if(head == HEADKS) {
      readExact(f,&nbLoadedWalk,sizeof(uint64_t));
      ::printf("ReadHeader: %s is a compressed kangaroo only file [2^%.2f kangaroos]\n",fileName.c_str(),log2((double)nbLoadedWalk));
    } else if(head==HEADW) {
      ::printf("ReadHeader: %s is a work file, kangaroo only file expected\n",fileName.c_str());
    } else {
      ::printf("ReadHeader: %s Not a work file\n",fileName.c_str());
    }
    ::fclose(f);
    return NULL;
  }
  return f;
}

bool Kangaroo::LoadWorkPayload(const std::string& fileName,const char* label,uint32_t* version,FILE** fOut,WorkFilePayload& payload) {
  FILE* f = ReadHeader(fileName,version,HEADW);
  if(f == NULL) {
    return false;
  }

  if(!readWorkPayload(f,payload)) {
    ::printf("%s: Cannot read work payload from %s\n",label,fileName.c_str());
    ::fclose(f);
    return false;
  }

  if(!secp->EC(payload.key)) {
    ::printf("%s: key does not lie on elliptic curve\n",label);
    ::fclose(f);
    return false;
  }

  if(fOut != NULL) {
    *fOut = f;
  } else {
    ::fclose(f);
  }
  return true;
}

bool Kangaroo::LoadWork(string &fileName) {
  double t0 = Timer::getTick();
  ::printf("Loading: %s\n",fileName.c_str());
  if(!clientMode) {
    WorkFilePayload payload;
    fRead = NULL;
    if(!LoadWorkPayload(fileName,"LoadWork",NULL,&fRead,payload))
      return false;
    keysToSearch.clear();
    if(initDPSize < 0) initDPSize = payload.dpSize;
    rangeStart = payload.rangeStart;
    rangeEnd = payload.rangeEnd;
    offsetCount = payload.totalCount;
    offsetTime = payload.totalTime;
    keysToSearch.push_back(payload.key);
    ::printf("Start:%s\n",rangeStart.GetBase16().c_str());
    ::printf("Stop :%s\n",rangeEnd.GetBase16().c_str());
    ::printf("Keys :%d\n",(int)keysToSearch.size());

    // Read hashTable
    hashTable.LoadTable(fRead);
  } else {
    // In client mode, config come from the server, file has only kangaroo
    fRead = NULL;
    fRead = ReadHeader(fileName,NULL,HEADK);
    if(fRead == NULL)
      return false;
  }
  // Read number of walk
  readExact(fRead,&nbLoadedWalk,sizeof(uint64_t));
  double t1 = Timer::getTick();
  ::printf("LoadWork: [HashTable %s] [%s]\n",hashTable.GetSizeInfo().c_str(),GetTimeStr(t1 - t0).c_str());
  return true;
}

void Kangaroo::FetchWalks(uint64_t nbWalk,Int *x,Int *y,Int *d) {
  // Read Kangaroos
  int64_t n = 0;
  ::printf("Fetch kangaroos: %.0f\n",(double)nbWalk);
  for(n = 0; n < (int64_t)nbWalk && nbLoadedWalk>0; n++) {
    readInt256(fRead,x[n]);
    readInt256(fRead,y[n]);
    readInt256(fRead,d[n]);
    nbLoadedWalk--;
  }
  if(n<(int64_t)nbWalk) {
    int64_t empty = nbWalk - n;
    // Fill empty kanagaroo
    CreateHerd((int)empty,&(x[n]),&(y[n]),&(d[n]),TAME);
  }
}

void Kangaroo::FetchWalks(uint64_t nbWalk,std::vector<int256_t>& kangs,Int* x,Int* y,Int* d) {
  uint64_t n = 0;
  uint64_t avail = (nbWalk<kangs.size())?nbWalk:kangs.size();

  if(avail > 0) {
    vector<Int> dists;
    vector<Point> Sp;
    dists.reserve(avail);
    Sp.reserve(avail);
    Point Z;
    Z.Clear();

    for(n = 0; n < avail; n++) {

      Int dist;
      uint32_t type;
      HashTable::CalcDistAndType(kangs[n],&dist,&type);
      dists.push_back(dist);

    }
    vector<Point> P = secp->ComputePublicKeys(dists);
    for(n = 0; n < avail; n++) {
      if(n % 2 == TAME) {
        Sp.push_back(Z);
      }
      else {
        Sp.push_back(keyToSearch);
      }
    }
    vector<Point> S = secp->AddDirect(Sp,P);
    for(n = 0; n < avail; n++) {
      x[n].Set(&S[n].x);
      y[n].Set(&S[n].y);
      d[n].Set(&dists[n]);
      nbLoadedWalk--;
    }
    kangs.erase(kangs.begin(),kangs.begin() + avail);
  }

  if(avail < nbWalk) {
    int64_t empty = nbWalk - avail;
    // Fill empty kanagaroo
    CreateHerd((int)empty,&(x[n]),&(y[n]),&(d[n]),TAME);
  }
}

void Kangaroo::FectchKangaroos(TH_PARAM *threads) {
  double sFetch = Timer::getTick();
  // From server
  vector<int256_t> kangs;
  if(saveKangarooByServer) {
    ::printf("FectchKangaroosFromServer");
    if(!GetKangaroosFromServer(workFile,kangs))
      ::exit(0);
    ::printf("Done\n");
    nbLoadedWalk = kangs.size();
  }

  // Fetch input kangaroo from file (if any)
  if(nbLoadedWalk>0) {
    ::printf("Restoring");
    uint64_t nbSaved = nbLoadedWalk;
    uint64_t created = 0;

    // Fetch loaded walk
    for(int i = 0; i < nbCPUThread; i++) {
      threads[i].px.resize(CPU_GRP_SIZE);
      threads[i].py.resize(CPU_GRP_SIZE);
      threads[i].distance.resize(CPU_GRP_SIZE);
      if(!saveKangarooByServer)
        FetchWalks(CPU_GRP_SIZE,threads[i].px.data(),threads[i].py.data(),threads[i].distance.data());
      else
        FetchWalks(CPU_GRP_SIZE,kangs,threads[i].px.data(),threads[i].py.data(),threads[i].distance.data());
    }
    ::printf("Done\n");
#ifdef WITHGPU
    for(int i = 0; i < nbGPUThread; i++) {
      ::printf(".");
      int id = nbCPUThread + i;
      uint64_t n = threads[id].nbKangaroo;
      threads[id].px.resize(n);
      threads[id].py.resize(n);
      threads[id].distance.resize(n);
      if(!saveKangarooByServer)
          FetchWalks(n,
            threads[id].px.data(),
            threads[id].py.data(),
            threads[id].distance.data());
      else
          FetchWalks(n,kangs,
            threads[id].px.data(),
            threads[id].py.data(),
            threads[id].distance.data());
    }
#endif
    double eFetch = Timer::getTick();
    if(nbLoadedWalk != 0) {
      ::printf("FectchKangaroos: Warning %.0f unhandled kangaroos !\n",(double)nbLoadedWalk);
    }
    if(nbSaved<totalRW)
      created = totalRW - nbSaved;
    ::printf("FectchKangaroos: [2^%.2f kangaroos loaded] [%.0f created] [%s]\n",log2((double)nbSaved),(double)created,GetTimeStr(eFetch - sFetch).c_str());
  }
  // Close input file
  if(fRead) fclose(fRead);
}

bool Kangaroo::SaveHeader(string fileName,FILE* f,int type,uint64_t totalCount,double totalTime) {
  // Header
  uint32_t head = type;
  uint32_t version = 0;
  if(::fwrite(&head,sizeof(uint32_t),1,f) != 1) {
    ::printf("SaveHeader: Cannot write to %s\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    return false;
  }
  ::fwrite(&version,sizeof(uint32_t),1,f);

  if(type==HEADW) {
    WorkFilePayload payload;
    payload.dpSize = dpSize;
    payload.rangeStart = rangeStart;
    payload.rangeEnd = rangeEnd;
    payload.key = keysToSearch[keyIdx];
    payload.totalCount = totalCount;
    payload.totalTime = totalTime;
    if(!writeWorkPayload(f,payload)) {
      ::printf("SaveHeader: Cannot write payload to %s\n",fileName.c_str());
      ::printf("%s\n",::strerror(errno));
      return false;
    }
  }
  return true;
}

void  Kangaroo::SaveWork(string fileName,FILE *f,int type,uint64_t totalCount,double totalTime) {
  ::printf("\nSaveWork: %s",fileName.c_str());
  // Header
  if(!SaveHeader(fileName,f,type,totalCount,totalTime))
    return;
  // Save hash table
  hashTable.SaveTable(f);
}

void Kangaroo::SaveServerWork() {
  saveRequest = true;
  double t0 = Timer::getTick();
  string fileName = workFile;

  if(splitWorkfile)
    fileName = workFile + "_" + Timer::getTS();
  FILE *f = fopen(fileName.c_str(),"wb");
  if(f == NULL) {
    ::printf("\nSaveWork: Cannot open %s for writing\n",fileName.c_str());
    ::printf("%s\n",::strerror(errno));
    saveRequest = false;
    return;
  }

  SaveWork(fileName,f,HEADW,0,0);
  uint64_t totalWalk = 0;
  ::fwrite(&totalWalk,sizeof(uint64_t),1,f);
  uint64_t size = FTell(f);
  fclose(f);

  if(splitWorkfile)
    hashTable.Reset();
  double t1 = Timer::getTick();
  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  ::printf("done [%.1f MB] [%s] %s",(double)size / (1024.0*1024.0),GetTimeStr(t1 - t0).c_str(),ctimeBuff);
  saveRequest = false;
}

void Kangaroo::SaveWork(uint64_t totalCount,double totalTime,TH_PARAM *threads,int nbThread) {
  uint64_t totalWalk = 0;
  uint64_t size;
  LOCK(saveMutex);
  double t0 = Timer::getTick();
  // Wait that all threads blocks before saving works
  saveRequest = true;
  int timeout = wtimeout;
  while(!isWaiting(threads) && timeout>0) {
    Timer::SleepMillis(50);
    timeout -= 50;
  }
  if(timeout<=0) {
    // Thread blocked or ended !
    if(!endOfSearch)
      ::printf("\nSaveWork timeout !\n");
    UNLOCK(saveMutex);
    return;
  }
  string fileName = workFile;
  if(splitWorkfile)
    fileName = workFile + "_" + Timer::getTS();
  // Save
  FILE* f = NULL;
  if(!saveKangarooByServer) {
    f = fopen(fileName.c_str(),"wb");
    if(f == NULL) {
      ::printf("\nSaveWork: Cannot open %s for writing\n",fileName.c_str());
      ::printf("%s\n",::strerror(errno));
      UNLOCK(saveMutex);
      return;
    }
  }
  if (clientMode) {
    if(saveKangarooByServer) {
      ::printf("\nSaveWork (Kangaroo->Server): %s",fileName.c_str());
      vector<int256_t> kangs;
      for(int i = 0; i < nbThread; i++)
        totalWalk += threads[i].nbKangaroo;
      kangs.reserve(totalWalk);

      for(int i = 0; i < nbThread; i++) {
        int256_t X;
        int256_t D;
        uint64_t h;
        for(uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
          HashTable::Convert(&threads[i].px[n],&threads[i].distance[n],n%2,&h,&X,&D);
          kangs.push_back(D);
        }
      }
      SendKangaroosToServer(fileName,kangs);
      size = kangs.size()*16 + 16;
      goto end;

    } else {
      SaveHeader(fileName,f,HEADK,totalCount,totalTime);
      ::printf("\nSaveWork (Kangaroo): %s",fileName.c_str());
    }
  } else {
    SaveWork(fileName,f,HEADW,totalCount,totalTime);
  }

  if(saveKangaroo) {
    // Save kangaroos
    for(int i = 0; i < nbThread; i++)
      totalWalk += threads[i].nbKangaroo;
    ::fwrite(&totalWalk,sizeof(uint64_t),1,f);

    uint64_t point = totalWalk / 16;
    uint64_t pointPrint = 0;

    for(int i = 0; i < nbThread; i++) {
      for(uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
        writeInt256(f,threads[i].px[n]);
        writeInt256(f,threads[i].py[n]);
        writeInt256(f,threads[i].distance[n]);
        pointPrint++;
        if(pointPrint>point) {
          ::printf(".");
          pointPrint = 0;
        }
      }
    }

  } else {
    ::fwrite(&totalWalk,sizeof(uint64_t),1,f);
  }
  
  size = FTell(f);
  fclose(f);
  if(splitWorkfile)
    hashTable.Reset();

  // Unblock threads
end:
  saveRequest = false;
  UNLOCK(saveMutex);
  double t1 = Timer::getTick();
  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  ::printf("done [%.1f MB] [%s] %s",(double)size/(1024.0*1024.0),GetTimeStr(t1 - t0).c_str(),ctimeBuff);
}

void Kangaroo::WorkInfo(std::string &fName) {
  int isDir = IsDir(fName);
  if(isDir<0)
    return;

  string fileName = fName;
  if(isDir)
    fileName = fName + "/header";

  ::printf("Loading: %s\n",fileName.c_str());

  uint32_t version;
  WorkFilePayload payload;
  FILE *f1 = NULL;
  if(!LoadWorkPayload(fileName,"WorkInfo",&version,&f1,payload))
    return;

  // Read hashTable
  if(isDir) {
    for(int i = 0; i < MERGE_PART; i++) {
      FILE* f = OpenPart(fName,"rb",i);
      hashTable.SeekNbItem(f,i * H_PER_PART,(i + 1) * H_PER_PART);
      fclose(f);
    }
  } else {
    hashTable.SeekNbItem(f1);
  }

  ::printf("Version   : %d\n",version);
  ::printf("DP bits   : %d\n",payload.dpSize);
  ::printf("Start     : %s\n",payload.rangeStart.GetBase16().c_str());
  ::printf("Stop      : %s\n",payload.rangeEnd.GetBase16().c_str());
  ::printf("Key       : %s\n",secp->GetPublicKeyHex(true,payload.key).c_str());
  ::printf("Count     : %" PRId64 " 2^%.3f\n",payload.totalCount,log2(payload.totalCount));
  ::printf("Time      : %s\n",GetTimeStr(payload.totalTime).c_str());
  hashTable.PrintInfo();
  readExact(f1,&nbLoadedWalk,sizeof(uint64_t));
  ::printf("Kangaroos : %" PRId64 " 2^%.3f\n",nbLoadedWalk,log2(nbLoadedWalk));
  fclose(f1);
}
