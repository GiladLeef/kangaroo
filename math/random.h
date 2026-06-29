#ifndef RANDOM_H
#define RANDOM_H

#include <cstdint>
#include <mutex>
#include <random>

namespace rng {

inline std::mt19937 &engine() {
  static std::mt19937 rng{0x600DCAFEU};
  return rng;
}

inline std::mutex &engineMutex() {
  static std::mutex mutex;
  return mutex;
}

inline void seed(uint32_t value) {
  std::lock_guard<std::mutex> lock(engineMutex());
  engine().seed(static_cast<std::mt19937::result_type>(value));
}

inline uint32_t next32() {
  std::lock_guard<std::mutex> lock(engineMutex());
  return engine()();
}

inline double nextDouble() {
  std::lock_guard<std::mutex> lock(engineMutex());
  return std::generate_canonical<double, 53>(engine());
}

}  // namespace rng

inline void rseed(unsigned long seed) {
  rng::seed(static_cast<uint32_t>(seed));
}

inline unsigned long rndl() {
  return static_cast<unsigned long>(rng::next32());
}

inline double rnd() {
  return rng::nextDouble();
}

#endif  // RANDOM_H
