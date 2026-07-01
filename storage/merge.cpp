#include "kangaroo.h"
#include <fstream>
#include "intgroup.h"
#include "timer.h"
#include "workfile.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <dirent.h>
#include <pthread.h>
#include <sys/stat.h>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace workfile;
namespace fs = std::filesystem;

namespace {

bool SameWorkParameters(const char* label,uint32_t v1,uint32_t v2,Int& rs1,Int& re1,Int& rs2,Int& re2,Point& k1,Point& k2) {
  if(v1 != v2) {
    ::printf("%s: cannot merge workfile of different version\n",label);
    return false;
  }

  if(!rs1.IsEqual(&rs2) || !re1.IsEqual(&re2)) {
    ::printf("%s: File range differs\n",label);
    ::printf("RS1: %s\n",rs1.GetBase16().c_str());
    ::printf("RE1: %s\n",re1.GetBase16().c_str());
    ::printf("RS2: %s\n",rs2.GetBase16().c_str());
    ::printf("RE2: %s\n",re2.GetBase16().c_str());
    return false;
  }

  if(!k1.equals(k2)) {
    ::printf("%s: key differs, multiple keys not yet supported\n",label);
    return false;
  }

  return true;
}

}  // namespace

string Kangaroo::GetPartName(const std::string& partName,int i,bool tmpPart) {
  std::ostringstream out;
  out << partName << "/part" << std::setw(3) << std::setfill('0') << i;
  if(tmpPart) {
    out << ".tmp";
  }
  return out.str();

}

FILE * Kangaroo::OpenPart(const std::string& partName,char *mode,int i,bool tmpPart) {
  string fName = GetPartName(partName,i,tmpPart);
  FILE* f = fopen(fName.c_str(),mode);
  if(f == NULL) {
    ::printf("OpenPart: Cannot open %s for mode %s\n",fName.c_str(),mode);
    ::printf("%s\n",::strerror(errno));
  }
  return f;
}

void Kangaroo::CreateEmptyPartWork(std::string& partName) {
    try {
        if (fs::exists(partName)) {
            return;
        }

        fs::create_directory(partName);

        string hName = partName + "/header";
        FileHandle f = openFile(hName, "wb");
        if (!f) {
            ::printf("CreateEmptyPartWork: Cannot open %s for writing\n", hName.c_str());
            ::printf("%s\n", ::strerror(errno));
            return;
        }

        for (int i = 0; i < MERGE_PART; i++) {
            FileHandle f(OpenPart(partName, "wb", i));
            if (!f)
                return;

            for (int j = 0; j < H_PER_PART; j++) {
                uint32_t z = 0;
                writeValue(f, z);
                writeValue(f, z);
            }
        }

        ::printf("CreateEmptyPartWork %s done\n", partName.c_str());
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Filesystem error: " << ex.what() << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
    } catch (...) {
        std::cerr << "Unknown error occurred\n";
    }
}

bool Kangaroo::MergePartition(TH_PARAM* p) {
  uint32_t part = p->hStart;
  const string& p1Name = p->part1Name;
  const string& p2Name = p->part2Name;

  FileHandle f1(OpenPart(p1Name,"rb",part,false));
  if(!f1) return false;
  FileHandle f2(OpenPart(p2Name,"rb",part,false));
  if(!f2) return false;
  FileHandle f(OpenPart(p1Name,"wb",part,true));
  if(!f) return false;

  uint32_t hStart = part * (HASH_SIZE / MERGE_PART);
  uint32_t hStop = (part +1) * (HASH_SIZE / MERGE_PART);

  uint32_t hDP;
  uint32_t hDuplicate;
  Int d1;
  uint32_t type1;
  Int d2;
  uint32_t type2;

  for(uint32_t h = hStart; h < hStop && !endOfSearch; h++) {

    int mStatus = HashTable::MergeH(h,f1.get(),f2.get(),f.get(),&hDP,&hDuplicate,&d1,&type1,&d2,&type2);
    switch(mStatus) {
    case ADD_OK:
      break;
    case ADD_COLLISION:
      CollisionCheck(&d1,type1,&d2,type2);
      break;
    }

    p->hStop += hDP;
    collisionInSameHerd += hDuplicate;

  }

  string oldName = GetPartName(p1Name,part,true);
  string newName = GetPartName(p1Name,part,false);
  if(!endOfSearch) {
    fs::remove(newName);
    fs::rename(oldName,newName);
  } else {
    fs::remove(oldName);
  }

  return true;

}

extern void* _mergeThread(void* lpParam);

void* _mergePartThread(void* lpParam) {
  TH_PARAM* p = (TH_PARAM*)lpParam;
  p->obj->MergePartition(p);
  p->isRunning = false;
  return 0;
}

bool Kangaroo::MergeWorkPartPart(std::string& part1Name,std::string& part2Name) {
  double t0;
  double t1;
  uint32_t v1;
  uint32_t v2;

  t0 = Timer::getTick();

  string file1 = part1Name + "/header";
  bool partIsEmpty = IsEmpty(file1);
  string file2 = part2Name + "/header";
  if( IsEmpty(file2) ) {
    ::printf("MergeWorkPartPart: partition #2 is empty\n");
    return true;
  }

  WorkFilePayload payload1;
  uint64_t count1 = 0;
  double time1 = 0;

  if(!partIsEmpty) {
    FileHandle f1;
    if(!LoadWorkPayload(file1,"MergeWorkPartPart",&v1,&f1,payload1))
      return true;
    count1 = payload1.totalCount;
    time1 = payload1.totalTime;
  }

  WorkFilePayload payload2;
  uint64_t count2 = 0;
  double time2 = 0;

  FileHandle f2;
  if(!LoadWorkPayload(file2,"MergeWorkPartPart",&v2,&f2,payload2)) {
    return true;
  }
  count2 = payload2.totalCount;
  time2 = payload2.totalTime;

  if(!partIsEmpty) {

    if(!SameWorkParameters("MergeWorkPartPart",v1,v2,payload1.rangeStart,payload1.rangeEnd,payload2.rangeStart,payload2.rangeEnd,payload1.key,payload2.key)) {
      return true;
    }

  } else {

    payload1 = payload2;

  }
  ::printf("%s: [DP%d]\n",part1Name.c_str(),payload1.dpSize);
  ::printf("%s: [DP%d]\n",part2Name.c_str(),payload2.dpSize);

  endOfSearch = false;
  if(!SetSearchContext(payload1.rangeStart,payload1.rangeEnd,payload1.key,"MergeWorkPartPart")) {
    return true;
  }

  FileHandle f = openFile(file1, "wb");
  if(!f) {
    ::printf("MergeWorkPart: Cannot open %s for writing\n",file1.c_str());
    ::printf("%s\n",::strerror(errno));
    return true;
  }
  dpSize = (payload1.dpSize < payload2.dpSize) ? payload1.dpSize : payload2.dpSize;
  if(!SaveHeader(file1,f.get(),HEADW,count1 + count2,time1 + time2)) {
    return true;
  }


  int nbCore = Timer::getCoreNumber();
  int l2 = (int)log2(nbCore);
  int nbThread = (int)pow(2.0,l2);
  if(nbThread > 16) nbThread = 16;

  setvbuf(stdout,NULL,_IONBF,0);

  ::printf("Thread: %d\n",nbThread);
  ::printf("Merging");

  std::vector<TH_PARAM> params(nbThread);
  std::vector<THREAD_HANDLE> thHandles(nbThread);
  uint64_t nbDP = 0;

  for(int p = 0; p < MERGE_PART && !endOfSearch; p+=nbThread) {

    printf(".");

    for(int i = 0; i < nbThread; i++) {
      params[i].threadId = i;
      params[i].isRunning = true;
      params[i].hStart = p+i;
      params[i].hStop = 0;
      params[i].part1Name = part1Name;
      params[i].part2Name = part2Name;
      thHandles[i] = LaunchThread(_mergePartThread,&params[i]);
    }

    JoinThreads(thHandles.data(),nbThread);

    for(int i = 0; i < nbThread; i++) {
      nbDP += params[i].hStop;
    }

  }

  t1 = Timer::getTick();

  if(!endOfSearch) {
    ::printf("Done [2^%.3f DP][%s]\n",log2((double)nbDP),GetTimeStr(t1 - t0).c_str());
    
  } else {
    ::printf("Dead kangaroo: %" PRId64 "\n",collisionInSameHerd);
    ::printf("Total f1+f2: DP count 2^%.2f\n",log2((double)nbDP));
    return true;
  }

  ::printf("Dead kangaroo: %" PRId64 "\n",collisionInSameHerd);
  ::printf("Total f1+f2: DP count 2^%.2f\n",log2((double)nbDP));
  return false;
}

bool Kangaroo::FillEmptyPartFromFile(std::string& partName,std::string& fileName,bool printStat) {
  double t0;
  double t1;
  uint32_t v1;
  WorkFilePayload payload1;

  t0 = Timer::getTick();

  FileHandle f1;
  if(!LoadWorkPayload(fileName,"FillEmptyPartFromFile",&v1,&f1,payload1))
    return true;

  dpSize = payload1.dpSize;
  keysToSearch.clear();
  keysToSearch.push_back(payload1.key);
  keyIdx = 0;
  collisionInSameHerd = 0;
  rangeStart.Set(&payload1.rangeStart);
  rangeEnd.Set(&payload1.rangeEnd);
  InitRange();
  InitSearchKey();

  string file1 = partName + "/header";
  FileHandle f = openFile(file1, "wb");
  if(!f) {
    ::printf("FillEmptyPartFromFile: Cannot open %s for writing\n",file1.c_str());
    ::printf("%s\n",::strerror(errno));
    return true;
  }
  if(!SaveHeader(file1,f.get(),HEADW,payload1.totalCount,payload1.totalTime)) {
    return true;
  }

  ::printf("Part %s: [DP%d]\n",partName.c_str(),payload1.dpSize);
  ::printf("File %s: [DP%d]\n",fileName.c_str(),payload1.dpSize);

  uint64_t nbDP = 0;
  ::printf("Filling");

  for(int p = 0; p < MERGE_PART; p++) {

    if(p % (MERGE_PART / 64) == 0) ::printf(".");

    FileHandle f(OpenPart(partName,"wb",p,false));
    uint32_t hStart = p * (HASH_SIZE / MERGE_PART);
    uint32_t hStop = (p + 1) * (HASH_SIZE / MERGE_PART);

    uint32_t nbItem;
    uint32_t maxItem;
    unsigned char buff[32];

    for(uint32_t h= hStart;h<hStop;h++) {
      readValue(f1, nbItem);
      readValue(f1, maxItem);
      writeValue(f, nbItem);
      writeValue(f, maxItem);
      for(uint32_t i=0;i<nbItem;i++) {
        ::fread(&buff,32,1,f1.get());
        ::fwrite(&buff,32,1,f.get());
      }
      nbDP += nbItem;
    }
  }

  t1 = Timer::getTick();
  ::printf("Done [2^%.3f DP][%s]\n",log2((double)nbDP),GetTimeStr(t1 - t0).c_str());
  
  return false;
}

bool Kangaroo::MergeWorkPart(std::string& partName,std::string& file2,bool printStat) {
  double t0;
  double t1;
  uint32_t v1;
  uint32_t v2;

  setvbuf(stdout,NULL,_IONBF,0);

  t0 = Timer::getTick();

  string file1 = partName + "/header";
  if(IsEmpty(file1))
    return FillEmptyPartFromFile(partName,file2,printStat);

  WorkFilePayload payload1;

  FileHandle f1;
  if(!LoadWorkPayload(file1,"MergeWorkPart",&v1,&f1,payload1))
    return true;

  WorkFilePayload payload2;
  FileHandle f2;
  if(!LoadWorkPayload(file2,"MergeWorkPart",&v2,&f2,payload2)) {
    return true;
  }

  if(!SameWorkParameters("MergeWorkPart",v1,v2,payload1.rangeStart,payload1.rangeEnd,payload2.rangeStart,payload2.rangeEnd,payload1.key,payload2.key)) {
    return true;
  }

  ::printf("Part %s: [DP%d]\n",partName.c_str(),payload1.dpSize);
  ::printf("File %s: [DP%d]\n",file2.c_str(),payload2.dpSize);

  endOfSearch = false;

  if(!SetSearchContext(payload1.rangeStart,payload1.rangeEnd,payload1.key,"MergeWorkPart")) {
    return true;
  }

  t0 = Timer::getTick();

  ::printf("Merging");

  FileHandle f = openFile(file1, "wb");
  if(!f) {
    ::printf("MergeWorkPart: Cannot open %s for writing\n",file1.c_str());
    ::printf("%s\n",::strerror(errno));
    return true;
  }
  dpSize = (payload1.dpSize < payload2.dpSize) ? payload1.dpSize : payload2.dpSize;
  if(!SaveHeader(file1,f.get(),HEADW,payload1.totalCount + payload2.totalCount,payload1.totalTime + payload2.totalTime)) {
    return true;
  }

  uint64_t nbDP = 0;
  uint32_t hDP;
  uint32_t hDuplicate;
  Int d1;
  uint32_t type1;
  Int d2;
  uint32_t type2;

  for(int part = 0; part < MERGE_PART && !endOfSearch; part++) {

    if(part % (MERGE_PART / 64) == 0) ::printf(".");

    uint32_t hStart = part * (HASH_SIZE / MERGE_PART);
    uint32_t hStop = (part + 1) * (HASH_SIZE / MERGE_PART);

    FileHandle f1p(OpenPart(partName,"rb",part));
    FileHandle fp(OpenPart(partName,"wb",part,true));

    for(uint32_t h = hStart; h < hStop && !endOfSearch; h++) {

      int mStatus = HashTable::MergeH(h,f1p.get(),f2.get(),fp.get(),&hDP,&hDuplicate,&d1,&type1,&d2,&type2);
      switch(mStatus) {
      case ADD_OK:
        break;
      case ADD_COLLISION:
        CollisionCheck(&d1,type1,&d2,type2);
        break;
      }

      nbDP += hDP;
      collisionInSameHerd += hDuplicate;

    }

    string oldName = GetPartName(partName,part,true);
    string newName = GetPartName(partName,part,false);
    if(!endOfSearch) {
      fs::remove(newName);
      fs::rename(oldName,newName);
    } else {
      fs::remove(oldName);
    }
  }

  t1 = Timer::getTick();

  if(!endOfSearch) {
    ::printf("Done [2^%.3f DP][%s]\n",log2((double)nbDP),GetTimeStr(t1 - t0).c_str());
    
  } else {
    ::printf("Dead kangaroo: %" PRId64 "\n",collisionInSameHerd);
    ::printf("Total f1+f2: DP count 2^%.2f\n",log2((double)nbDP));
    return true;
  }

  if(printStat) {
    ::printf("Dead kangaroo: %" PRId64 "\n",collisionInSameHerd);
    ::printf("Total f1+f2: DP count 2^%.2f\n",log2((double)nbDP));
  }

  return false;
}

bool Kangaroo::MergeWork(std::string& file1,std::string& file2,std::string& dest,bool printStat) {
  if(IsDir(file1) && IsDir(file2)) {
    return MergeWorkPartPart(file1,file2);
  }

  if(IsDir(file1)) {
    return MergeWorkPart(file1,file2,true);
  }

  if(dest.empty()) {
    ::printf("MergeWork: destination argument missing\n");
    return true;
  }

  double t0 = Timer::getTick();
  double t1;
  uint32_t v1;
  uint32_t v2;

  WorkFilePayload payload1;
  FileHandle f1;
  if(!LoadWorkPayload(file1,"MergeWork",&v1,&f1,payload1))
    return true;

  WorkFilePayload payload2;
  FileHandle f2;
  if(!LoadWorkPayload(file2,"MergeWork",&v2,&f2,payload2)) {
    return true;
  }

  if(!SameWorkParameters("MergeWork",v1,v2,payload1.rangeStart,payload1.rangeEnd,payload2.rangeStart,payload2.rangeEnd,payload1.key,payload2.key)) {
    return true;
  }

  ::printf("%s: [DP%d]\n",file1.c_str(),payload1.dpSize);
  ::printf("%s: [DP%d]\n",file2.c_str(),payload2.dpSize);

  endOfSearch = false;
  if(!SetSearchContext(payload1.rangeStart,payload1.rangeEnd,payload1.key,"MergeWork")) {
    return true;
  }

  FileHandle f = openFile(dest, "wb");
  if(!f) {
    ::printf("MergeWork: Cannot open %s for writing\n",dest.c_str());
    ::printf("%s\n",::strerror(errno));
    return true;
  }

  dpSize = (payload1.dpSize < payload2.dpSize) ? payload1.dpSize : payload2.dpSize;
  if(!SaveHeader(dest,f.get(),HEADW,payload1.totalCount + payload2.totalCount,payload1.totalTime + payload2.totalTime)) {
    return true;
  }

  ::printf("Merging");
  if(printStat) {
    setvbuf(stdout,NULL,_IONBF,0);
  }

  uint64_t nbDP = 0;
  uint32_t hDP;
  uint32_t hDuplicate;
  Int d1;
  uint32_t type1;
  Int d2;
  uint32_t type2;

  for(uint32_t h = 0; h < HASH_SIZE && !endOfSearch; h++) {
    if(printStat && (h % (HASH_SIZE / 64) == 0)) {
      ::printf(".");
    }

    int mStatus = HashTable::MergeH(h,f1.get(),f2.get(),f.get(),&hDP,&hDuplicate,&d1,&type1,&d2,&type2);
    switch(mStatus) {
    case ADD_OK:
      break;
    case ADD_COLLISION:
      CollisionCheck(&d1,type1,&d2,type2);
      break;
    }

    nbDP += hDP;
    collisionInSameHerd += hDuplicate;
  }

  uint64_t nbWalk1 = 0;
  uint64_t nbWalk2 = 0;
  readValue(f1, nbWalk1);
  readValue(f2, nbWalk2);
  uint64_t totalWalk = nbWalk1 + nbWalk2;
  writeValue(f, totalWalk);

  t1 = Timer::getTick();

  if(!endOfSearch) {
    ::printf("Done [2^%.3f DP][%s]\n",log2((double)nbDP),GetTimeStr(t1 - t0).c_str());
  } else {
    ::printf("Dead kangaroo: %" PRId64 "\n",collisionInSameHerd);
    ::printf("Total f1+f2: DP count 2^%.2f\n",log2((double)nbDP));
    return true;
  }

  if(printStat) {
    ::printf("Dead kangaroo: %" PRId64 "\n",collisionInSameHerd);
    ::printf("Total f1+f2: DP count 2^%.2f\n",log2((double)nbDP));
  }

  return false;
}

void Kangaroo::MergeDir(std::string& dirName,std::string& dest) {
  struct File {
    std::string name;
    uint64_t size;
  };

  std::vector<File> listFiles;
  for(const auto& entry : fs::directory_iterator(dirName)) {
    if(fs::is_regular_file(entry.path())) {
      uint32_t version = 0;
      FileHandle f(ReadHeader(entry.path().string(),&version,HEADW));
      if(f) {
        File e{entry.path().string(),0};
        fseeko(f.get(),0,SEEK_END);
        e.size = (uint64_t)ftello(f.get());
        listFiles.push_back(e);
      }
    }
  }

  std::sort(listFiles.begin(),listFiles.end(),[](const File& lhs,const File& rhs) {
    return lhs.size > rhs.size;
  });

  int lgth = (int)listFiles.size();
  if(IsDir(dest) == 1) {
    bool end = false;
    for(int i = 0; i < lgth && !end; i++) {
      std::cout << "\n## File #" << i + 1 << "/" << lgth << std::endl;
      end = MergeWorkPart(dest,listFiles[i].name,i == lgth - 1);
    }
  } else {
    if(listFiles.size() < 2) {
      std::cout << "MergeDir: less than 2 work files in the directory" << std::endl;
      return;
    }

    std::cout << "\n## File #1/" << lgth - 1 << std::endl;
    bool end = MergeWork(listFiles[0].name,listFiles[1].name,dest,lgth == 2);
    for(int i = 2; i < lgth && !end; i++) {
      std::cout << "\n## File #" << i << "/" << lgth - 1 << std::endl;
      end = MergeWork(dest,listFiles[i].name,dest,i == lgth - 1);
    }
  }
}
