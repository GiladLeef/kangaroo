#ifndef WORK_FILE_H
#define WORK_FILE_H

#include "int.h"
#include "point.h"
#include <cstdint>
#include <cstdio>

namespace workfile {

struct WorkFilePayload {
  uint32_t dpSize = 0;
  Int rangeStart;
  Int rangeEnd;
  Point key;
  uint64_t totalCount = 0;
  double totalTime = 0.0;
};

inline bool readExact(FILE* f, void* dst, size_t size, size_t count = 1) {
  return ::fread(dst, size, count, f) == count;
}

inline bool writeExact(FILE* f, const void* src, size_t size, size_t count = 1) {
  return ::fwrite(src, size, count, f) == count;
}

inline bool readInt256(FILE* f, Int& value) {
  if (!readExact(f, &value.bits64, 32)) {
    return false;
  }
  value.bits64[4] = 0;
  return true;
}

inline bool writeInt256(FILE* f, const Int& value) {
  return writeExact(f, &value.bits64, 32);
}

inline bool readPointAffine(FILE* f, Point& value) {
  if (!readInt256(f, value.x) || !readInt256(f, value.y)) {
    return false;
  }
  value.z.SetInt32(1);
  return true;
}

inline bool writePointAffine(FILE* f, const Point& value) {
  return writeInt256(f, value.x) && writeInt256(f, value.y);
}

inline bool readWorkPayload(FILE* f, WorkFilePayload& payload) {
  return readExact(f, &payload.dpSize, sizeof(uint32_t)) &&
         readInt256(f, payload.rangeStart) &&
         readInt256(f, payload.rangeEnd) &&
         readPointAffine(f, payload.key) &&
         readExact(f, &payload.totalCount, sizeof(uint64_t)) &&
         readExact(f, &payload.totalTime, sizeof(double));
}

inline bool writeWorkPayload(FILE* f, const WorkFilePayload& payload) {
  return writeExact(f, &payload.dpSize, sizeof(uint32_t)) &&
         writeInt256(f, payload.rangeStart) &&
         writeInt256(f, payload.rangeEnd) &&
         writePointAffine(f, payload.key) &&
         writeExact(f, &payload.totalCount, sizeof(uint64_t)) &&
         writeExact(f, &payload.totalTime, sizeof(double));
}

}  // namespace workfile

#endif  // WORK_FILE_H
