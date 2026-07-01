#ifndef WORK_FILE_H
#define WORK_FILE_H

#include "int.h"
#include "point.h"
#include <cstdint>
#include <istream>
#include <ostream>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>

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

#endif  // WORK_FILE_H
