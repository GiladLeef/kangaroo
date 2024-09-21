#include "Kangaroo.h"
#include <filesystem>
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <iostream>
#include <cmath> // For log2
#include <algorithm> // For std::sort

namespace fs = std::filesystem;

typedef struct File {
    std::string name;
    uint64_t size;
} File;

bool sortBySize(const File& lhs, const File& rhs) { return lhs.size > rhs.size; }

bool isRegularFile(const std::string& filePath) {
    return fs::is_regular_file(filePath);
}
bool Kangaroo::MergeWork(std::string& file1, std::string& file2, std::string& dest, bool printStat) {

  if (IsDir(file1) && IsDir(file2)) {
    return MergeWorkPartPart(file1, file2);
  }

  if (IsDir(file1)) {
    return MergeWorkPart(file1, file2, true);
  }

  if (dest.length() == 0) {
    ::printf("MergeWork: destination argument missing\n");
    return true;
  }

  double t0;
  double t1;
  uint32_t v1;
  uint32_t v2;

  t0 = Timer::getTick();

  FILE* f1 = ReadHeader(file1, &v1, HEADW);
  if (f1 == NULL)
    return false;

  uint32_t dp1;
  Point k1;
  uint64_t count1;
  double time1;
  Int RS1;
  Int RE1;

  // Read global param
  ::fread(&dp1, sizeof(uint32_t), 1, f1);
  ::fread(&RS1.bits64, 32, 1, f1); RS1.bits64[4] = 0;
  ::fread(&RE1.bits64, 32, 1, f1); RE1.bits64[4] = 0;
  ::fread(&k1.x.bits64, 32, 1, f1); k1.x.bits64[4] = 0;
  ::fread(&k1.y.bits64, 32, 1, f1); k1.y.bits64[4] = 0;
  ::fread(&count1, sizeof(uint64_t), 1, f1);
  ::fread(&time1, sizeof(double), 1, f1);

  k1.z.SetInt32(1);
  if (!secp->EC(k1)) {
    ::printf("MergeWork: key1 does not lie on elliptic curve\n");
    fclose(f1);
    return true;
  }
  return true;
}


void Kangaroo::MergeDir(std::string& dirName, std::string& dest) {
    std::vector<File> listFiles;

    for (const auto& entry : fs::directory_iterator(dirName)) {
        if (fs::is_regular_file(entry.path())) {
            uint32_t version;
            FILE *f = ReadHeader(entry.path().string(), &version, HEADW);
            if (f) {
                File e;
                e.name = entry.path().string();
                fseeko(f, 0, SEEK_END);
                e.size = (uint64_t)ftello(f);
                listFiles.push_back(e);
                fclose(f);
            }
        }
    }

    std::sort(listFiles.begin(), listFiles.end(), sortBySize);

    int lgth = (int)listFiles.size();

    if (IsDir(dest) == 1) {
        bool end = false;
        for (int i = 0; i < lgth && !end; i++) {
            std::cout << "\n## File #" << i + 1 << "/" << lgth << std::endl;
            end = MergeWorkPart(dest, listFiles[i].name, i == lgth - 1);
        }
    } else {
        if (listFiles.size() < 2) {
            std::cout << "MergeDir: less than 2 work files in the directory" << std::endl;
            return;
        }

        int i = 0;
        std::cout << "\n## File #1/" << lgth - 1 << std::endl;
        bool end = MergeWork(listFiles[0].name, listFiles[1].name, dest, lgth == 2);
        for (int i = 2; i < lgth && !end; i++) {
            std::cout << "\n## File #" << i << "/" << lgth - 1 << std::endl;
            end = MergeWork(dest, listFiles[i].name, dest, i == lgth - 1);
        }
    }
}
