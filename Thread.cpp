#include "Kangaroo.h"
#include "Timer.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <pthread.h>

using namespace std;

THREAD_HANDLE Kangaroo::LaunchThread(void *(*func) (void *), TH_PARAM *p) {
  THREAD_HANDLE h;
  p->obj = this;
  pthread_create(&h, NULL, func, (void*)(p));
  return h;
}
void  Kangaroo::JoinThreads(THREAD_HANDLE *handles, int nbThread) {
  for (int i = 0; i < nbThread; i++)
    pthread_join(handles[i], NULL);
}

bool Kangaroo::isAlive(TH_PARAM *p) {
  bool isAlive = false;
  int total = nbCPUThread + nbGPUThread;
  for(int i=0;i<total;i++)
    isAlive = isAlive || p[i].isRunning;
  return isAlive;
}

bool Kangaroo::hasStarted(TH_PARAM *p) {
  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
    hasStarted = hasStarted && p[i].hasStarted;
  return hasStarted;
}

bool Kangaroo::isWaiting(TH_PARAM *p) {
  bool isWaiting = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
    isWaiting = isWaiting && p[i].isWaiting;
  return isWaiting;
}

uint64_t Kangaroo::getGPUCount() {
  uint64_t count = 0;
  for(int i = 0; i<nbGPUThread; i++)
    count += counters[0x80L + i];
  return count;
}

uint64_t Kangaroo::getCPUCount() {
  uint64_t count = 0;
  for(int i=0;i<nbCPUThread;i++)
    count += counters[i];
  return count;
}

string Kangaroo::GetTimeStr(double dTime) {

  char tmp[256];

  double nbDay = dTime / 86400.0;
  if (nbDay >= 1) {

    double nbYear = nbDay / 365.0;
    if (nbYear > 1) {
      if (nbYear < 5)
        sprintf(tmp, "%.1fy", nbYear);
      else
        sprintf(tmp, "%gy", nbYear);
    } else {
      sprintf(tmp, "%.1fd", nbDay);
    }

  } else {

    int iTime = (int)dTime;
    int nbHour = (int)((iTime % 86400) / 3600);
    int nbMin = (int)(((iTime % 86400) % 3600) / 60);
    int nbSec = (int)(iTime % 60);

    if (nbHour == 0) {
      if (nbMin == 0) {
        sprintf(tmp, "%02ds", nbSec);
      } else {
        sprintf(tmp, "%02d:%02d", nbMin, nbSec);
      }
    } else {
      sprintf(tmp, "%02d:%02d:%02d", nbHour, nbMin, nbSec);
    }

  }

  return string(tmp);

}

// Wait for end of server and dispay stats
void Kangaroo::ProcessServer() {
    pthread_mutex_init(&ghMutex, NULL);
    setvbuf(stdout, NULL, _IONBF, 0);

    double t0 = Timer::getTick();
    double lastSave = 0;

    while (!endOfSearch) {
        double t1 = Timer::getTick();
        LOCK(ghMutex);
        localCache.assign(recvDP.begin(), recvDP.end());
        recvDP.clear();
        UNLOCK(ghMutex);

        for (const auto& dp : localCache) {
            for (int j = 0; j < dp.nbDP && !endOfSearch; j++) {
                uint64_t h = dp.dp[j].h;
                if (!AddToTable(h, &dp.dp[j].x, &dp.dp[j].d)) {
                    collisionInSameHerd++;
                }
            }
            free(dp.dp);
        }

        double elapsedTime = Timer::getTick() - t1;
        double toSleep = std::max(0.0, SEND_PERIOD - elapsedTime);
        Timer::SleepMillis(static_cast<uint32_t>(toSleep * 1000.0));

        if (!endOfSearch) {
            printf("\r[Client %d][Kang 2^%.2f][DP Count 2^%.2f/2^%.2f][Dead %.0f][%s][%s]  ",
                   connectedClient,
                   log2((double)totalRW),
                   log2((double)hashTable.GetNbItem()),
                   log2(expectedNbOp / pow(2.0, dpSize)),
                   (double)collisionInSameHerd,
                   GetTimeStr(Timer::getTick() - startTime).c_str(),
                   hashTable.GetSizeInfo().c_str());
        }

        if (!workFile.empty() && !endOfSearch) {
            if ((Timer::getTick() - lastSave) > saveWorkPeriod) {
                SaveServerWork();
                lastSave = Timer::getTick();
            }
        }
    }
}

// Wait for end of threads and display stats
void Kangaroo::Process(TH_PARAM *params,std::string unit) {

  double t0;
  double t1;
  uint64_t count;
  uint64_t lastCount = 0;
  uint64_t gpuCount = 0;
  uint64_t lastGPUCount = 0;
  double avgKeyRate = 0.0;
  double avgGpuKeyRate = 0.0;
  double lastSave = 0;

  setvbuf(stdout, NULL, _IONBF, 0);

  // Key rate smoothing filter
#define FILTER_SIZE 8
  double lastkeyRate[FILTER_SIZE];
  double lastGpukeyRate[FILTER_SIZE];
  uint32_t filterPos = 0;

  double keyRate = 0.0;
  double gpuKeyRate = 0.0;

  memset(lastkeyRate,0,sizeof(lastkeyRate));
  memset(lastGpukeyRate,0,sizeof(lastkeyRate));

  // Wait that all threads have started
  while(!hasStarted(params))
    Timer::SleepMillis(5);

  t0 = Timer::getTick();
  startTime = t0;
  lastGPUCount = getGPUCount();
  lastCount = getCPUCount() + gpuCount;

  while(isAlive(params)) {

    int delay = 2000;
    while(isAlive(params) && delay > 0) {
      Timer::SleepMillis(50);
      delay -= 50;
    }

    gpuCount = getGPUCount();
    count = getCPUCount() + gpuCount;

    t1 = Timer::getTick();
    keyRate = (double)(count - lastCount) / (t1 - t0);
    gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
    lastkeyRate[filterPos%FILTER_SIZE] = keyRate;
    lastGpukeyRate[filterPos%FILTER_SIZE] = gpuKeyRate;
    filterPos++;

    // KeyRate smoothing
    uint32_t nbSample;
    for(nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
      avgKeyRate += lastkeyRate[nbSample];
      avgGpuKeyRate += lastGpukeyRate[nbSample];
    }
    avgKeyRate /= (double)(nbSample);
    avgGpuKeyRate /= (double)(nbSample);
    double expectedTime = expectedNbOp / avgKeyRate;

    // Display stats
    if(isAlive(params) && !endOfSearch) {
      if(clientMode) {
        printf("\r[%.2f %s][GPU %.2f %s][Count 2^%.2f][%s][Server %6s]  ",
          avgKeyRate / 1000000.0,unit.c_str(),
          avgGpuKeyRate / 1000000.0,unit.c_str(),
          log2((double)count + offsetCount),
          GetTimeStr(t1 - startTime + offsetTime).c_str(),
          serverStatus.c_str()
          );
      } else {
        printf("\r[%.2f %s][GPU %.2f %s][Count 2^%.2f][Dead %.0f][%s (Avg %s)][%s]  ",
          avgKeyRate / 1000000.0,unit.c_str(),
          avgGpuKeyRate / 1000000.0,unit.c_str(),
          log2((double)count + offsetCount),
          (double)collisionInSameHerd,
          GetTimeStr(t1 - startTime + offsetTime).c_str(),GetTimeStr(expectedTime).c_str(),
          hashTable.GetSizeInfo().c_str()
        );
      }

    }

    // Save request
    if(workFile.length() > 0 && !endOfSearch) {
      if((t1 - lastSave) > saveWorkPeriod) {
        SaveWork(count + offsetCount,t1 - startTime + offsetTime,params,nbCPUThread + nbGPUThread);
        lastSave = t1;
      }
    }

    // Abort
    if(!clientMode && maxStep>0.0) {
      double max = expectedNbOp * maxStep; 
      if( (double)count > max ) {
        ::printf("\nKey#%2d [XX]Pub:  0x%s \n",keyIdx,secp->GetPublicKeyHex(true,keysToSearch[keyIdx]).c_str());
        ::printf("       Aborted !\n");
        endOfSearch = true;
        Timer::SleepMillis(1000);
      }
    }
    lastCount = count;
    lastGPUCount = gpuCount;
    t0 = t1;
  }

  count = getCPUCount() + getGPUCount();
  t1 = Timer::getTick();

  if(!endOfSearch) {
    printf("\r[%.2f %s][GPU %.2f %s][Cnt 2^%.2f][%s]  ",
      avgKeyRate / 1000000.0,unit.c_str(),
      avgGpuKeyRate / 1000000.0,unit.c_str(),
      log2((double)count),
      GetTimeStr(t1 - startTime).c_str()
      );
  }
}