#ifndef TIMERH
#define TIMERH

#include <time.h>
#include <string>

class Timer {

public:
  static void Init();
  static double get_tick();
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

#endif // TIMERH