#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <signal.h>
#include <pthread.h>
#include <unordered_map>
#include <regex>
#include <chrono>

using namespace std;

static SOCKET serverSock = 0;

// Common part
#define MAX_CLIENT 256
#define WAIT_FOR_READ  1
#define WAIT_FOR_WRITE 2

#define SERVER_VERSION 1

#define SERVER_HEADER 0x67DEDDC1

#define KANG_PER_BLOCK 1024

// Commands
#define SERVER_GETCONFIG 0
#define SERVER_STATUS    1
#define SERVER_SENDDP    2
#define SERVER_SETKNB    3
#define SERVER_SAVEKANG  4
#define SERVER_LOADKANG  5
#define SERVER_RESETDEAD  'R'
#define SERVER_GETPOOLSTATS 'P'

// Status
#define SERVER_OK            0
#define SERVER_END           1
#define SERVER_BACKUP        2


#define close_socket(s) close(s)

string GetNetworkError() {

  return string(strerror(errno));

}

#define GET(name,s,b,bl,t)  if( (nbRead=Read(s,(char *)(b),bl,t))<0 ) { ::printf("\nReadError(" name "): %s\n",lastError.c_str()); isConnected = false; close_socket(s); return false; }
#define PUT(name,s,b,bl,t)  if( (nbWrite=Write(s,(char *)(b),bl,t))<0 ) { ::printf("\nWriteError(" name "): %s\n",lastError.c_str()); isConnected = false; close_socket(s); return false; }
#define GETFREE(name,s,b,bl,t,x)  if( (nbRead=Read(s,(char *)(b),bl,t))<0 ) { ::printf("\nReadError(" name "): %s\n",lastError.c_str()); isConnected = false; ::free(x); close_socket(s); return false; }
#define PUTFREE(name,s,b,bl,t,x)  if( (nbWrite=Write(s,(char *)(b),bl,t))<0 ) { ::printf("\nWriteError(" name "): %s\n",lastError.c_str()); isConnected = false; ::free(x); close_socket(s); return false; }

void sig_handler(int signo) {
  if(signo == SIGINT) {
    ::printf("\nTerminated\n");
    if(serverSock>0) close_socket(serverSock);
    exit(0);
  }
}

int Kangaroo::WaitFor(SOCKET sock,int timeout,int mode) {
  fd_set fdset;
  fd_set *rd = NULL,*wr = NULL;
  struct timeval tmout;
  int result;

  FD_ZERO(&fdset);
  FD_SET(sock,&fdset);
  if(mode == WAIT_FOR_READ)
    rd = &fdset;
  if(mode == WAIT_FOR_WRITE)
    wr = &fdset;

  tmout.tv_sec = (int)(timeout / 1000);
  tmout.tv_usec = (int)(timeout % 1000) * 1000;

  do
    result = select((int)sock + 1,rd,wr,NULL,&tmout);
  while(result < 0 && errno == EINTR);

  if(result == 0) {
    lastError = "The operation timed out";
  } else if(result < 0) {
    lastError = GetNetworkError();
    return 0;
  }
  return result;
}

int Kangaroo::Write(SOCKET sock,char *buf,int bufsize,int timeout) {
  int total_written = 0;
  int written = 0;

  while(bufsize > 0)
  {
    // Wait
    if(!WaitFor(sock,timeout,WAIT_FOR_WRITE))
      return -1;
    // Write
    do
      written = send(sock,buf,bufsize,0);
    while(written == -1 && errno == EINTR);
    if(written <= 0)
      break;

    buf += written;
    total_written += written;
    bufsize -= written;
  }
  if(written < 0) {
    lastError = GetNetworkError();
    return -1;
  }
  if(bufsize != 0) {
    lastError = "Failed to send entire buffer";
    return -1;
  }
  return total_written;
}

int Kangaroo::Read(SOCKET sock,char *buf,int bufsize,int timeout) { // Timeout in millisec
  int rd = 0;
  int total_read = 0;

  while( bufsize>0 ) {
    // Wait
    if(!WaitFor(sock,timeout,WAIT_FOR_READ)) {
      return -1;
    }
    // Read
    do
      rd = recv(sock,buf,bufsize,0);

    while(rd == -1 && errno == EINTR);
    if( rd <= 0 )
      break;

    buf += rd;
    total_read += rd;
    bufsize -= rd;
  }
  if(rd < 0) {
    lastError = GetNetworkError();
    return -1;
  }
  if(rd == 0) {
    lastError = "Connection closed";
    return -1;
  }
  return total_read;
}

// Check if string is a valid Bitcoin P2PK address
bool Kangaroo::IsValidBitcoinAddress(const std::string& address) {
  // Simple regex check for P2PK format (starting with '1')
  std::regex p2pkRegex("^1[a-zA-Z0-9]{25,34}$");
  return std::regex_match(address, p2pkRegex);
}

// Update client statistics for pool mode
void Kangaroo::UpdateClientStats(const std::string& address, uint32_t dpCount, const std::string& clientInfo) {
  LOCK(poolStatsMutex);
  
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::system_clock::to_time_t(now);
  
  if (clientStats.find(address) == clientStats.end()) {
    // New client
    CLIENT_STATS stats;
    stats.address = address;
    stats.dpCount = dpCount;
    stats.lastSeen = timestamp;
    stats.clientInfo = clientInfo;
    clientStats[address] = stats;
  } else {
    // Update existing client
    clientStats[address].dpCount += dpCount;
    clientStats[address].lastSeen = timestamp;
    clientStats[address].clientInfo = clientInfo;
  }
  
  totalPoolDP += dpCount;
  
  // Save stats to file periodically
  SavePoolStats();
  
  UNLOCK(poolStatsMutex);
}

// Get a copy of the client stats for the UI
std::unordered_map<std::string, CLIENT_STATS> Kangaroo::GetClientStats() {
  LOCK(poolStatsMutex);
  auto stats = clientStats;
  UNLOCK(poolStatsMutex);
  return stats;
}

// Get total DP count for the pool
uint64_t Kangaroo::GetTotalDP() {
  LOCK(poolStatsMutex);
  auto total = totalPoolDP;
  UNLOCK(poolStatsMutex);
  return total;
}

// Save pool statistics to a JSON file
bool Kangaroo::SavePoolStats() {
  std::ofstream statsFile("poolstats.json");
  if (!statsFile.is_open()) {
    ::printf("\nCannot open poolstats.json for writing\n");
    return false;
  }
  
  statsFile << "{\n";
  statsFile << "  \"totalDP\": " << totalPoolDP << ",\n";
  statsFile << "  \"clients\": [\n";
  
  bool first = true;
  for (const auto& client : clientStats) {
    if (!first) {
      statsFile << ",\n";
    }
    first = false;
    
    time_t lastSeen = client.second.lastSeen;
    char timeBuffer[30];
    std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&lastSeen));
    
    statsFile << "    {\n";
    statsFile << "      \"address\": \"" << client.second.address << "\",\n";
    statsFile << "      \"dpCount\": " << client.second.dpCount << ",\n";
    statsFile << "      \"lastSeen\": \"" << timeBuffer << "\",\n";
    statsFile << "      \"clientInfo\": \"" << client.second.clientInfo << "\",\n";
    statsFile << "      \"percentage\": " << (totalPoolDP > 0 ? (double)client.second.dpCount / totalPoolDP * 100.0 : 0.0) << "\n";
    statsFile << "    }";
  }
  
  statsFile << "\n  ]\n}";
  statsFile.close();
  return true;
}

// ------------------------------------------------------------------------------------------------------
// Server code
// ------------------------------------------------------------------------------------------------------

// Server status
int32_t Kangaroo::GetServerStatus() {
  if(endOfSearch) {
    return SERVER_END;
  }
  if(saveRequest) {
    return SERVER_BACKUP;
  }
  return SERVER_OK;
}

#define CLIENT_ABORT() \
::printf("\nClosing connection with %s\n",p->clientInfo); \
close_socket(p->clientSock); \
return false;


// Server request handler for pool mode
bool Kangaroo::HandlePoolRequest(TH_PARAM *p) {
  char cmdBuff;
  uint32_t version = SERVER_VERSION;
  int nbRead;
  int nbWrite;
  int32_t state;
  
  // For pool mode, we need to get the Bitcoin address first
  uint32_t addrLength;
  char bitcoinAddress[128];
  
  GET("AddressLength", p->clientSock, &addrLength, sizeof(uint32_t), ntimeout);
  
  if (addrLength >= 128) {
    ::printf("\nBitcoin address too long (MAX=127) from %s\n", p->clientInfo);
    CLIENT_ABORT();
  }
  
  GET("BitcoinAddress", p->clientSock, &bitcoinAddress, addrLength, ntimeout);
  bitcoinAddress[addrLength] = 0;
  
  // Verify the Bitcoin address
  if (!IsValidBitcoinAddress(bitcoinAddress)) {
    ::printf("\nInvalid Bitcoin address '%s' from %s\n", bitcoinAddress, p->clientInfo);
    CLIENT_ABORT();
  }
  
  // Store the Bitcoin address
  p->bitcoinAddress = strdup(bitcoinAddress);
  ::printf("\nNew pool connection from %s with address %s\n", p->clientInfo, p->bitcoinAddress);
  
  while (p->isRunning) {
    // Wait for command (1h timeout)
    nbRead = Read(p->clientSock, (char *)(&cmdBuff), 1, (int)(CLIENT_TIMEOUT*1000.0));
    if (nbRead <= 0) {
      CLIENT_ABORT();
    }
    
    switch (cmdBuff) {
      case SERVER_GETCONFIG: {
        // Send config to the client
        PUT("Version", p->clientSock, &version, sizeof(uint32_t), ntimeout);
        PUT("RangeStart", p->clientSock, rangeStart.bits64, 32, ntimeout);
        PUT("RangeEnd", p->clientSock, rangeEnd.bits64, 32, ntimeout);
        PUT("KeyX", p->clientSock, keysToSearch[keyIdx].x.bits64, 32, ntimeout);
        PUT("KeyY", p->clientSock, keysToSearch[keyIdx].y.bits64, 32, ntimeout);
        PUT("DP", p->clientSock, &initDPSize, sizeof(int32_t), ntimeout);
      } break;

      case SERVER_SETKNB: {
        GET("nbKangaroo", p->clientSock, &p->nbKangaroo, sizeof(uint64_t), ntimeout);
        totalRW += p->nbKangaroo;
      } break;
      
      case SERVER_RESETDEAD: {
        char response[5];
        collisionInSameHerd = 0;
        GET("flush", p->clientSock, &response, 2, ntimeout);
        sprintf(response, "OK\n");
        PUT("resp", p->clientSock, &response, 3, ntimeout);
      } break;

      case SERVER_LOADKANG: {
        Int checkSum;
        Int K;
        uint64_t nbKangaroo = 0;
        uint32_t strSize;
        char fileName[256];
        int256_t* KBuff;
        uint32_t nbK;
        uint32_t header = HEADKS;
        uint32_t version = 0;

        GET("fileNameLenght", p->clientSock, &strSize, sizeof(uint32_t), ntimeout);
        if (strSize >= 256) {
          ::printf("\nFileName too long (MAX=256) %s\n", p->clientInfo);
          CLIENT_ABORT();
        }

        GET("fileName", p->clientSock, &fileName, strSize, ntimeout);
        fileName[strSize] = 0;
        FILE* f = fopen(fileName, "rb");
        if (f == NULL) {
          // No backup
          ::printf("LoadKang: Cannot open %s for reading\n", fileName);
          ::printf("%s\n", ::strerror(errno));
          PUT("nbKangaroo", p->clientSock, &nbKangaroo, sizeof(uint64_t), ntimeout);
          break;
        }

        if (::fread(&header, sizeof(uint32_t), 1, f) != 1) {
          ::printf("LoadKang: Cannot read from %s\n", fileName);
          ::printf("%s\n", ::strerror(errno));
          ::fclose(f);
          CLIENT_ABORT();
        }

        if (header != HEADKS) {
          ::printf("LoadKang: %s Not a compressed kangaroo file\n", fileName);
          ::printf("%s\n", ::strerror(errno));
          ::fclose(f);
          CLIENT_ABORT();
        }

        ::fread(&version, sizeof(uint32_t), 1, f);
        ::fread(&nbKangaroo, sizeof(uint64_t), 1, f);

        PUT("nbKangaroo", p->clientSock, &nbKangaroo, sizeof(uint64_t), ntimeout);

        checkSum.SetInt32(0);
        KBuff = (int256_t*)malloc(KANG_PER_BLOCK * sizeof(int256_t));

        while (nbKangaroo > 0) {
          if (nbKangaroo > KANG_PER_BLOCK) {
            nbK = KANG_PER_BLOCK;
          } else {
            nbK = (uint32_t)nbKangaroo;
          }

          for (uint32_t k = 0; k < nbK; k++) {
            ::fread(&KBuff[k], 16, 1, f);
            // Checksum
            K.SetInt32(0);
            K.bits64[3] = KBuff[k].i64[3];
            K.bits64[2] = KBuff[k].i64[2];
            K.bits64[1] = KBuff[k].i64[1];
            K.bits64[0] = KBuff[k].i64[0];
            checkSum.Add(&K);
          }

          PUTFREE("packet", p->clientSock, KBuff, nbK * 16, ntimeout, KBuff);
          nbKangaroo -= nbK;
        }
        free(KBuff);
        PUT("checkSum", p->clientSock, checkSum.bits64, 32, ntimeout);
        ::fclose(f);
      } break;

      case SERVER_SAVEKANG: {
        Int checkSum;
        Int K;
        uint64_t nbKangaroo;
        uint32_t fileNameSize;
        char fileNameTmp[264];
        char fileName[256];
        int256_t *KBuff;
        uint32_t nbK;
        uint32_t header = HEADKS;
        uint32_t version = 0;

        GET("fileNameLenght", p->clientSock, &fileNameSize, sizeof(uint32_t), ntimeout);
        if (fileNameSize >= 256) {
          ::printf("\nFileName too long (MAX=256) %s\n", p->clientInfo);
          CLIENT_ABORT();
        }

        GET("fileName", p->clientSock, &fileName, fileNameSize, ntimeout);
        fileName[fileNameSize] = 0;
        GET("nbKangaroo", p->clientSock, &nbKangaroo, sizeof(uint64_t), ntimeout);

        strcpy(fileNameTmp, fileName);
        strcat(fileNameTmp, ".tmp");

        FILE* f = fopen(fileNameTmp, "wb");
        if (f == NULL) {
          ::printf("\nCannot open %s for writing\n", fileNameTmp);
          ::printf("%s\n", ::strerror(errno));
          CLIENT_ABORT();
        }

        if (::fwrite(&header, sizeof(uint32_t), 1, f) != 1) {
          ::printf("\nCannot write to %s\n", fileNameTmp);
          ::printf("%s\n", ::strerror(errno));
          ::fclose(f);
          CLIENT_ABORT();
        }
        ::fwrite(&version, sizeof(uint32_t), 1, f);
        ::fwrite(&nbKangaroo, sizeof(uint64_t), 1, f);
        
        checkSum.SetInt32(0);
        KBuff = (int256_t *)malloc(KANG_PER_BLOCK*sizeof(int256_t));
        
        while (nbKangaroo > 0) {
          if (nbKangaroo > KANG_PER_BLOCK) {
            nbK = KANG_PER_BLOCK;
          } else {
            nbK = (uint32_t)nbKangaroo;
          }

          GETFREE("packet", p->clientSock, KBuff, nbK * 16, ntimeout, KBuff);
          
          for (uint32_t k = 0; k < nbK; k++) {
            ::fwrite(&KBuff[k], 16, 1, f);
            // Checksum
            K.SetInt32(0);
            K.bits64[3] = KBuff[k].i64[3];
            K.bits64[2] = KBuff[k].i64[2];
            K.bits64[1] = KBuff[k].i64[1];
            K.bits64[0] = KBuff[k].i64[0];
            checkSum.Add(&K);
          }
          nbKangaroo -= nbK;
        }

        free(KBuff);
        ::fclose(f);

        K.SetInt32(0);
        GET("checksum", p->clientSock, K.bits64, 32, ntimeout);

        if (!K.IsEqual(&checkSum)) {
          ::printf("\nWarning, Kangaroo backup wrong checksum %s\n", fileName);
        } else {
          remove(fileName);
          rename(fileNameTmp, fileName);
        }
      } break;

      case SERVER_STATUS: {
        state = GetServerStatus();
        PUT("Status", p->clientSock, &state, sizeof(int32_t), ntimeout);
      } break;
      
      case SERVER_GETPOOLSTATS: {
        // Send pool statistics to the client
        uint64_t totalDP = GetTotalDP();
        PUT("TotalDP", p->clientSock, &totalDP, sizeof(uint64_t), ntimeout);
        
        std::string address(p->bitcoinAddress);
        uint64_t clientDP = 0;
        double percentage = 0.0;
        
        auto stats = GetClientStats();
        auto it = stats.find(address);
        if (it != stats.end()) {
          clientDP = it->second.dpCount;
          percentage = totalDP > 0 ? (double)clientDP / totalDP * 100.0 : 0.0;
        }
        
        PUT("ClientDP", p->clientSock, &clientDP, sizeof(uint64_t), ntimeout);
        PUT("Percentage", p->clientSock, &percentage, sizeof(double), ntimeout);
      } break;
      
      case SERVER_SENDDP: {
        DPHEADER head;
        GET("DPHeader", p->clientSock, &head, sizeof(DPHEADER), ntimeout);
        if (head.header != SERVER_HEADER) {
          ::printf("\nUnexpected DP header from %s\n", p->clientInfo);
          CLIENT_ABORT();
        }

        if (head.nbDP == 0) {
          ::printf("\nUnexpected number of DP [%d] from %s\n", head.nbDP, p->clientInfo);
          CLIENT_ABORT();
        } else {
          DP *dp = (DP *)malloc(sizeof(DP) * head.nbDP);
          GETFREE("DP", p->clientSock, dp, sizeof(DP) * head.nbDP, ntimeout, dp);
          state = GetServerStatus();
          PUTFREE("Status", p->clientSock, &state, sizeof(int32_t), ntimeout, dp);
          
          if (nbRead != sizeof(DP) * head.nbDP) {
            ::printf("\nUnexpected DP size from %s [nbDP=%d, Got %d, Expected %d]\n",
                p->clientInfo, head.nbDP, nbRead, (int)(sizeof(DP) * head.nbDP));
            free(dp);
            CLIENT_ABORT();
          } else {
            // Enhanced security: validate each DP before counting
            uint32_t validDPs = 0;
            std::vector<DP> validatedDPs;
            
            for (uint32_t i = 0; i < head.nbDP; i++) {
              // Check 1: Verify DP is valid (has the required number of leading zeros)
              // The high bits of the x-coordinate should match our DP mask
              if (!IsDP(dp[i].x.i64[3])) {
                ::printf("Invalid DP from %s: DP #%d doesn't have required leading zeros\n", 
                         p->clientInfo, i);
                continue;
              }
              
              // Check 2: Verify this DP hasn't been submitted before (by any client)
              bool isDuplicate = false;
              LOCK(ghMutex);
              
              // Check against our processed DPs
              for (const auto& cache : recvDP) {
                for (uint32_t j = 0; j < cache.nbDP; j++) {
                  if (memcmp(&dp[i].x, &cache.dp[j].x, sizeof(int256_t)) == 0) {
                    isDuplicate = true;
                    break;
                  }
                }
                if (isDuplicate) break;
              }
              UNLOCK(ghMutex);
              
              if (isDuplicate) {
                ::printf("Duplicate DP from %s: DP #%d was already submitted\n", 
                         p->clientInfo, i);
                continue;
              }
              
              // This DP passed all checks, it's valid
              validDPs++;
              validatedDPs.push_back(dp[i]);
            }
            
            // If we found any valid DPs, update the client stats and add to processing
            if (validDPs > 0) {
              // Update client statistics with only the valid DP count
              UpdateClientStats(p->bitcoinAddress, validDPs, p->clientInfo);
              
              // Create a new array with only valid DPs
              DP *validDpArray = (DP *)malloc(sizeof(DP) * validDPs);
              for (uint32_t i = 0; i < validDPs; i++) {
                memcpy(&validDpArray[i], &validatedDPs[i], sizeof(DP));
              }
              
              // Add the validated DPs to processing queue
              LOCK(ghMutex);
              DP_CACHE dc;
              dc.nbDP = validDPs;
              dc.dp = validDpArray;
              recvDP.push_back(dc);
              UNLOCK(ghMutex);
              
              ::printf("Received %d valid DPs from %s (client reported %d)\n", 
                       validDPs, p->clientInfo, head.nbDP);
            } else {
              ::printf("No valid DPs from %s (client reported %d)\n", 
                       p->clientInfo, head.nbDP);
            }
            
            // Free the original DP array since we've created a new one with only valid DPs
            free(dp);
          }
        }
      } break;

      default:
        ::printf("\nUnexpected command [%d] from %s\n", cmdBuff, p->clientInfo);
        CLIENT_ABORT();
    }
  }

  close_socket(p->clientSock);
  return true;
}

// Server request handler
bool Kangaroo::HandleRequest(TH_PARAM *p) {

  char cmdBuff;
  uint32_t version = SERVER_VERSION;
  int nbRead;
  int nbWrite;
  int32_t state;

  while( p->isRunning ) {
    // Wait for command (1h timeout)
    nbRead = Read(p->clientSock,(char *)(&cmdBuff),1,(int)(CLIENT_TIMEOUT*1000.0));
    if(nbRead<=0) {
      CLIENT_ABORT();
    }
    
    switch(cmdBuff) {
    case SERVER_GETCONFIG: {
      ::printf("\nNew connection from %s\n",p->clientInfo);
      // Send config to the client
      PUT("Version",p->clientSock,&version,sizeof(uint32_t),ntimeout);
      PUT("RangeStart",p->clientSock,rangeStart.bits64,32,ntimeout);
      PUT("RangeEnd",p->clientSock,rangeEnd.bits64,32,ntimeout);
      PUT("KeyX",p->clientSock,keysToSearch[keyIdx].x.bits64,32,ntimeout);
      PUT("KeyY",p->clientSock,keysToSearch[keyIdx].y.bits64,32,ntimeout);
      PUT("DP",p->clientSock,&initDPSize,sizeof(int32_t),ntimeout);
    } break;

    case SERVER_SETKNB: {
      GET("nbKangaroo",p->clientSock,&p->nbKangaroo,sizeof(uint64_t),ntimeout);
      totalRW += p->nbKangaroo;
    } break;
      
    case SERVER_RESETDEAD: {
      char response[5];
      collisionInSameHerd = 0;
      GET("flush",p->clientSock,&response,2,ntimeout);
      sprintf(response,"OK\n");
      PUT("resp",p->clientSock,&response,3,ntimeout);
    } break;

    case SERVER_LOADKANG: {
      Int checkSum;
      Int K;
      uint64_t nbKangaroo = 0;
      uint32_t strSize;
      char fileName[256];
      int256_t* KBuff;
      uint32_t nbK;
      uint32_t header = HEADKS;
      uint32_t version = 0;

      GET("fileNameLenght",p->clientSock,&strSize,sizeof(uint32_t),ntimeout);
      if(strSize >= 256) {
        ::printf("\nFileName too long (MAX=256) %s\n",p->clientInfo);
        CLIENT_ABORT();
      }

      GET("fileName",p->clientSock,&fileName,strSize,ntimeout);
      fileName[strSize] = 0;
      FILE* f = fopen(fileName,"rb");
      if(f == NULL) {
        // No backup
        ::printf("LoadKang: Cannot open %s for reading\n",fileName);
        ::printf("%s\n",::strerror(errno));
        PUT("nbKangaroo",p->clientSock,&nbKangaroo,sizeof(uint64_t),ntimeout);
        break;
      }

      if(::fread(&header,sizeof(uint32_t),1,f) != 1) {
        ::printf("LoadKang: Cannot read from %s\n",fileName);
        ::printf("%s\n",::strerror(errno));
        ::fclose(f);
        CLIENT_ABORT();
      }

      if(header!=HEADKS) {
        ::printf("LoadKang: %s Not a compressed kangaroo file\n",fileName);
        ::printf("%s\n",::strerror(errno));
        ::fclose(f);
        CLIENT_ABORT();
      }

      ::fread(&version,sizeof(uint32_t),1,f);
      ::fread(&nbKangaroo,sizeof(uint64_t),1,f);

      PUT("nbKangaroo",p->clientSock,&nbKangaroo,sizeof(uint64_t),ntimeout);

      checkSum.SetInt32(0);
      KBuff = (int256_t*)malloc(KANG_PER_BLOCK * sizeof(int256_t));

      while(nbKangaroo > 0) {

        if(nbKangaroo > KANG_PER_BLOCK) {
          nbK = KANG_PER_BLOCK;
        }  else {
          nbK = (uint32_t)nbKangaroo;
        }

        for(uint32_t k = 0; k < nbK; k++) {
          ::fread(&KBuff[k],16,1,f);
          // Checksum
          K.SetInt32(0);
          K.bits64[3] = KBuff[k].i64[3];
          K.bits64[2] = KBuff[k].i64[2];
          K.bits64[1] = KBuff[k].i64[1];
          K.bits64[0] = KBuff[k].i64[0];
          checkSum.Add(&K);
        }

        PUTFREE("packet",p->clientSock,KBuff,nbK * 16,ntimeout,KBuff);
        nbKangaroo -= nbK;
      }
      free(KBuff);
      PUT("checkSum",p->clientSock,checkSum.bits64,32,ntimeout);
      ::fclose(f);
    } break;

    case SERVER_SAVEKANG: {
      Int checkSum;
      Int K;
      uint64_t nbKangaroo;
      uint32_t fileNameSize;
      char fileNameTmp[264];
      char fileName[256];
      int256_t *KBuff;
      uint32_t nbK;
      uint32_t header = HEADKS;
      uint32_t version = 0;

      GET("fileNameLenght",p->clientSock,&fileNameSize,sizeof(uint32_t),ntimeout);
      if(fileNameSize >= 256) {
        ::printf("\nFileName too long (MAX=256) %s\n",p->clientInfo);
        CLIENT_ABORT();
      }

      GET("fileName",p->clientSock,&fileName,fileNameSize,ntimeout);
      fileName[fileNameSize]=0;
      GET("nbKangaroo",p->clientSock,&nbKangaroo,sizeof(uint64_t),ntimeout);

      strcpy(fileNameTmp,fileName);
      strcat(fileNameTmp,".tmp");

      FILE* f = fopen(fileNameTmp,"wb");
      if(f == NULL) {
        ::printf("\nCannot open %s for writing\n",fileNameTmp);
        ::printf("%s\n",::strerror(errno));
        CLIENT_ABORT();
      }

      if(::fwrite(&header,sizeof(uint32_t),1,f) != 1) {
        ::printf("\nCannot write to %s\n",fileNameTmp);
        ::printf("%s\n",::strerror(errno));
        ::fclose(f);
        CLIENT_ABORT();
      }
      ::fwrite(&version,sizeof(uint32_t),1,f);
      ::fwrite(&nbKangaroo,sizeof(uint64_t),1,f);
      
      checkSum.SetInt32(0);
      KBuff = (int256_t *)malloc(KANG_PER_BLOCK*sizeof(int256_t));
      
      while(nbKangaroo>0) {
        if(nbKangaroo> KANG_PER_BLOCK) {
          nbK = KANG_PER_BLOCK;
        } else {
          nbK = (uint32_t)nbKangaroo;
        }

        GETFREE("packet",p->clientSock,KBuff,nbK * 16,ntimeout,KBuff);
        
        for(uint32_t k = 0; k < nbK; k++) {
          ::fwrite(&KBuff[k],16,1,f);
          // Checksum
          K.SetInt32(0);
          K.bits64[3] = KBuff[k].i64[3];
          K.bits64[2] = KBuff[k].i64[2];
          K.bits64[1] = KBuff[k].i64[1];
          K.bits64[0] = KBuff[k].i64[0];
          checkSum.Add(&K);
        }
        nbKangaroo -= nbK;
      }

      free(KBuff);
      ::fclose(f);

      K.SetInt32(0);
      GET("checksum",p->clientSock,K.bits64,32,ntimeout);

      if(!K.IsEqual(&checkSum)) {
        ::printf("\nWarning, Kangaroo backup wrong checksum %s\n",fileName);
      } else {
        remove(fileName);
        rename(fileNameTmp,fileName);
      }

    } break;

    case SERVER_STATUS: {

      state = GetServerStatus();
      PUT("Status",p->clientSock,&state,sizeof(int32_t),ntimeout);

    } break;
    
    case SERVER_SENDDP: {
      DPHEADER head;
      GET("DPHeader", p->clientSock, &head, sizeof(DPHEADER), ntimeout);
      if (head.header != SERVER_HEADER) {
          ::printf("\nUnexpected DP header from %s\n", p->clientInfo);
          CLIENT_ABORT();
      }

      if (head.nbDP == 0) {
          ::printf("\nUnexpected number of DP [%d] from %s\n", head.nbDP, p->clientInfo);
          CLIENT_ABORT();
      } else {
          DP *dp = (DP *)malloc(sizeof(DP) * head.nbDP);
          GETFREE("DP", p->clientSock, dp, sizeof(DP) * head.nbDP, ntimeout, dp);
          state = GetServerStatus();
          PUTFREE("Status", p->clientSock, &state, sizeof(int32_t), ntimeout, dp);
          
          if (nbRead != sizeof(DP) * head.nbDP) {
              ::printf("\nUnexpected DP size from %s [nbDP=%d, Got %d, Expected %d]\n",
                  p->clientInfo, head.nbDP, nbRead, (int)(sizeof(DP) * head.nbDP));
              free(dp);
              CLIENT_ABORT();
          } else {              
              LOCK(ghMutex);
              DP_CACHE dc;
              dc.nbDP = head.nbDP;
              dc.dp = dp;
              recvDP.push_back(dc);
              UNLOCK(ghMutex);
          }
      }
    } break;

    default:
      ::printf("\nUnexpected command [%d] from %s\n",cmdBuff,p->clientInfo);
      CLIENT_ABORT();
    }
  }

  close_socket(p->clientSock);
  return true;
}

void *_acceptThread(void *lpParam) {
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->AddConnectedClient();
  p->obj->HandleRequest(p);
  p->obj->RemoveConnectedClient();
  p->obj->RemoveConnectedKangaroo(p->nbKangaroo);
  p->isRunning = false;
  free(p->clientInfo);
  free(p);
  return 0;
}

void *_acceptPoolThread(void *lpParam) {
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->AddConnectedClient();
  p->obj->HandlePoolRequest(p);
  p->obj->RemoveConnectedClient();
  p->obj->RemoveConnectedKangaroo(p->nbKangaroo);
  p->isRunning = false;
  free(p->clientInfo);
  if (p->bitcoinAddress) {
    free(p->bitcoinAddress);
  }
  free(p);
  return 0;
}

void *_processServer(void *lpParam) {
  Kangaroo *obj = (Kangaroo *)lpParam;
  obj->ProcessServer();
  return 0;
}

void *_processPoolServer(void *lpParam) {
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->ProcessPoolServer();
  return 0;
}

// Process function for pool server
void Kangaroo::ProcessPoolServer() {
  ProcessServer(); // Use the same processing logic as regular server
}

// Pool server connections acceptor
void Kangaroo::AcceptPoolConnections(SOCKET server_soc) {
  SOCKET clientSock;
  ::printf("Kangaroo pool server is ready and listening to TCP port %d ...\n",port);
  
  while(true) {
    struct sockaddr_in client_add;
    socklen_t len = sizeof(sockaddr_in);
    if((clientSock = accept(server_soc,(struct sockaddr*)&client_add,&len)) < 0) {
      ::printf("Error: Invalid Socket returned by accept(): %s\n",GetNetworkError().c_str());
    } else {
      TH_PARAM *p = (TH_PARAM *)malloc(sizeof(TH_PARAM));
      ::memset(p,0,sizeof(TH_PARAM));
      char info[256];
      ::sprintf(info,"%s:%d",inet_ntoa(client_add.sin_addr),ntohs(client_add.sin_port));
      p->clientInfo = ::strdup(info);
      p->obj = this;
      p->isRunning = true;
      p->clientSock = clientSock;
      LaunchThread(_acceptPoolThread,p);
    }
  }
}

// Main server loop
void Kangaroo::AcceptConnections(SOCKET server_soc) {
  SOCKET clientSock;
  ::printf("Kangaroo server is ready and listening to TCP port %d ...\n",port);
  
  while(true) {
    struct sockaddr_in client_add;
    socklen_t len = sizeof(sockaddr_in);
    if((clientSock = accept(server_soc,(struct sockaddr*)&client_add,&len)) < 0) {
      ::printf("Error: Invalid Socket returned by accept(): %s\n",GetNetworkError().c_str());
    } else {
      TH_PARAM *p = (TH_PARAM *)malloc(sizeof(TH_PARAM));
      ::memset(p,0,sizeof(TH_PARAM));
      char info[256];
      ::sprintf(info,"%s:%d",inet_ntoa(client_add.sin_addr),ntohs(client_add.sin_port));
      p->clientInfo = ::strdup(info);
      p->obj = this;
      p->isRunning = true;
      p->clientSock = clientSock;
      LaunchThread(_acceptThread,p);
    }
  }
}

// Run pool server
void Kangaroo::RunPoolServer() {
  
  if(signal(SIGINT,sig_handler) == SIG_ERR)
    ::printf("\nWarning:can't install singal handler\n");

  // Set starting parameters
  InitRange();
  InitSearchKey();

  ComputeExpected((double)initDPSize,&expectedNbOp,&expectedMem);
  ::printf("Expected operations: 2^%.2f\n",log2(expectedNbOp));
  ::printf("Expected RAM usage: %.1fMB\n",expectedMem);

  if(initDPSize<0) {
    ::printf("Error: Pool server must be launched with a specified number of distinguished bits (-d)\n");
    exit(-1);
  }
  SetDP(initDPSize);

  // Initialize pool stats tracking
  pthread_mutex_init(&poolStatsMutex,NULL);
  totalPoolDP = 0;
  clientStats.clear();
  
  // Create the server socket
  struct sockaddr_in soc_addr;
  int serverSockOpt = 1;
  struct protoent *p;
  
  #ifdef WIN64
    WSADATA WSAData;
    if(WSAStartup(MAKEWORD(2,2),&WSAData)!=0) {
      printf("Error: Cannot start Windows socket\n");
      exit(-1);
    }
  #endif

  // Create socket
  if((serverSock = socket(AF_INET,SOCK_STREAM,0)) < 0) {
    ::printf("Error: Unable to create socket: %s\n",GetNetworkError().c_str());
    exit(-1);
  }

  if(setsockopt(serverSock,SOL_SOCKET,SO_REUSEADDR,(char *)&serverSockOpt,sizeof(serverSockOpt)) < 0) {
    ::printf("Error: setsockopt(SO_REUSEADDR): %s\n",GetNetworkError().c_str());
    close_socket(serverSock);
    exit(-1);
  }

  // Reuse address
  memset(&soc_addr,0,sizeof(soc_addr));
  soc_addr.sin_family = AF_INET;
  soc_addr.sin_addr.s_addr = INADDR_ANY;
  soc_addr.sin_port = htons(port);

  // Bind
  if(bind(serverSock,(struct sockaddr*)&soc_addr,sizeof(soc_addr))) {
    ::printf("Error: Unable to bind: %s\n",GetNetworkError().c_str());
    close_socket(serverSock);
    exit(-1);
  }

  // Listen
  if(listen(serverSock,5)) {
    ::printf("Error: Unable to listen: %s\n",GetNetworkError().c_str());
    close_socket(serverSock);
    exit(-1);
  }

  // Thread managment
  pthread_t processThread;

  // Process thread (DP processing)
  TH_PARAM *threadParam = (TH_PARAM *)malloc(sizeof(TH_PARAM));
  memset(threadParam, 0, sizeof(TH_PARAM));
  threadParam->obj = this;
  threadParam->isRunning = true;
  LaunchThread(_processPoolServer, threadParam);

  // Accept connections
  AcceptPoolConnections(serverSock);

}

// Starts the server
void Kangaroo::RunServer() {
  
  if(signal(SIGINT,sig_handler) == SIG_ERR)
    ::printf("\nWarning:can't install singal handler\n");

  // Set starting parameters
  InitRange();
  InitSearchKey();

  ComputeExpected((double)initDPSize,&expectedNbOp,&expectedMem);
  ::printf("Expected operations: 2^%.2f\n",log2(expectedNbOp));
  ::printf("Expected RAM usage: %.1fMB\n",expectedMem);

  if(initDPSize<0) {
    ::printf("Error: Server must be launched with a specified number of distinguished bits (-d)\n");
    exit(-1);
  }
  SetDP(initDPSize);

  if(saveKangaroo) {
    ::printf("Waring: Server does not support -ws, ignoring\n");
    saveKangaroo = false;
  }

  // Main thread of server (handle backup and collision check)
  LaunchThread(_processServer,(TH_PARAM *)this);
  Timer::SleepMillis(100);

  // Server stuff
  
  /* Create socket */
  serverSock = socket(AF_INET,SOCK_STREAM,0);

  if(serverSock<0) {
    ::printf("Error: Invalid socket : %s\n",GetNetworkError().c_str());
    exit(-1);
  }

  struct sockaddr_in soc_addr;

  /* Reuse Address */
  int32_t yes = 1;
  if(setsockopt(serverSock,SOL_SOCKET,SO_REUSEADDR,(char *)&yes,sizeof(yes)) < 0) {
    ::printf("Warning: Couldn't Reuse Address: %s\n",GetNetworkError().c_str());
  }
  memset(&soc_addr,0,sizeof(soc_addr));
  soc_addr.sin_family = AF_INET;
  soc_addr.sin_port = htons(port);
  soc_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if(bind(serverSock,(struct sockaddr*)&soc_addr,sizeof(soc_addr))) {
    ::printf("Error: Can not bind socket. Another server running?\n%s\n",GetNetworkError().c_str());
    exit(-1);
  }
  if(listen(serverSock,MAX_CLIENT)<0) {
    ::printf("Error: Can not listen to socket\n%s\n",GetNetworkError().c_str());
    exit(-1);
  }
  AcceptConnections(serverSock);
  return;
}

// ---------------------------------------------------------------------------------
// Client part
// ---------------------------------------------------------------------------------

// Connection to the server
bool Kangaroo::ConnectToServer(SOCKET *retSock) {

  lastError = "";
  // Resolve IP
  if(!hostInfo) {

    if(signal(SIGINT,sig_handler) == SIG_ERR)
      ::printf("\nWarning:can't install singal handler\n");

    struct hostent *host_info;
    host_info = gethostbyname(serverIp.c_str());
    if(host_info == NULL) {
      lastError = "Unknown host:" + serverIp;
      hostInfo = NULL;
      hostInfoLength = 0;
      return false;
    } else {
      hostInfoLength = host_info->h_length;
      hostInfo = (char *)malloc(hostInfoLength);
      ::memcpy(hostInfo,host_info->h_addr,hostInfoLength);
      hostAddrType = host_info->h_addrtype;
    }
  }

  struct sockaddr_in server;

  // Build TCP connection
  SOCKET sock = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
  if(sock < 0) {
    lastError = "Socket error: " + GetNetworkError();
    return false;
  }

  // Use non blocking socket
  if(fcntl(sock,F_SETFL,O_NONBLOCK) == -1) {
    lastError = "Cannot use non blocking socket, " + GetNetworkError();
    close_socket(sock);
    return false;
  }

  // Connect
  ::memset(&server,0,sizeof(sockaddr_in));
  server.sin_family = hostAddrType;
  ::memcpy((char*)&server.sin_addr,hostInfo,hostInfoLength);
  server.sin_port = htons(port);

  int connectStatus = connect(sock,(struct sockaddr *)&server,sizeof(server));

  if((connectStatus < 0) && (errno != EINPROGRESS)) {
    lastError = "Cannot connect to host: " + GetNetworkError();
    close_socket(sock);
    return false;
  }

  if(connectStatus<0) {

    // Wait for connection
    if(!WaitFor(sock,ntimeout,WAIT_FOR_WRITE)) {
      lastError = "Cannot connect, unreachable host " + serverIp;
      close_socket(sock);
      return false;
    }

    // Check connection completion
    int socket_err;
    socklen_t serrlen = sizeof(socket_err);
    if(getsockopt(sock,SOL_SOCKET,SO_ERROR,&socket_err,&serrlen) == -1) {
      lastError = "Cannot connect to host: " + GetNetworkError();
      close_socket(sock);
      return false;
    }

    if(socket_err != 0) {
      lastError = "Cannot connect to host: " + string(strerror(socket_err));
      close_socket(sock);
      return false;
    }
  }

  int on = 1;
  if(setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,
    (const char*)&on,sizeof(on)) == -1) {
    lastError = "Socket error: setsockopt error SO_REUSEADDR";
    close_socket(sock);
    return false;
  }

  int flag = 1;
  struct protoent *p;
  p = getprotobyname("tcp");
  if(setsockopt(sock,p->p_proto,TCP_NODELAY,(char *)&flag,sizeof(flag)) == -1) {
    lastError = "Socket error: setsockopt error TCP_NODELAY";
    close_socket(sock);
    return false;
  }

  *retSock = sock;
  return true;

}

// Wait while server is not ready
void Kangaroo::WaitForServer() {

  int nbRead;
  int nbWrite;
  int32_t status;
  bool ok = false;

  while(!ok) {
    
    // Wait for connection
    while(!isConnected) {
      serverStatus = "Disconnected";
      Timer::SleepMillis(1000);
      // Try to reconnect
      isConnected = ConnectToServer(&serverConn);

      if( isConnected ) {

        // Resend kangaroo number
        char cmd = SERVER_SETKNB;
        nbWrite = Write(serverConn,&cmd,1,ntimeout);
        if(nbWrite <= 0) {
          if(nbWrite < 0)
            ::printf("\nSendToServer(SetKNb): %s\n",lastError.c_str());
          serverStatus = "Not OK";
          close_socket(serverConn);
          isConnected = false;
        }
        nbWrite = Write(serverConn,(char *)&totalRW,sizeof(uint64_t),ntimeout);
        if(nbWrite <= 0) {
          if(nbWrite < 0)
            ::printf("\nSendToServer(SetKNb): %s\n",lastError.c_str());
          serverStatus = "Not OK";
          close_socket(serverConn);
          isConnected = false;
        }
      }
    }

    // Wait for ready
    while(isConnected && !ok) {

      char cmd = SERVER_STATUS;
      nbWrite = Write(serverConn,&cmd,1,ntimeout);
      if( nbWrite<=0 ) {

        if(nbWrite<0)
          ::printf("\nSendToServer(Status): %s\n",lastError.c_str()); 
        serverStatus = "Not OK";
        close_socket(serverConn);
        isConnected = false;

      } else {

        nbRead = Read(serverConn,(char *)(&status),sizeof(int32_t),ntimeout);
        if( nbRead<=0 ) {
          if(nbRead<0)
            ::printf("\nRecvFromServer(Status): %s\n",lastError.c_str()); 
          serverStatus = "Disconnected";
          close_socket(serverConn);
          isConnected = false;
        } else {

          switch(status) {
          case SERVER_OK:
            serverStatus = "Connected";
            ok = true;
            break;

          case SERVER_END:
            serverStatus = "END";
            endOfSearch = true;
            ok = true;
            break;

          case SERVER_BACKUP:
            serverStatus = "Backup";
            Timer::SleepMillis(1000);
            break;
          }
        }
      }
    }
  }
}

// Get Kangaroo from server
bool Kangaroo::GetKangaroosFromServer(std::string& fileName,std::vector<int256_t>& kangs) {

  int nbRead;
  int nbWrite;
  uint32_t fileNameSize = (uint32_t)fileName.length();
  uint64_t nbKangaroo = 0;
  uint32_t nbK;
  int256_t* KBuff;
  Int checkSum;
  WaitForServer();

  if(!endOfSearch) {
    char cmd = SERVER_LOADKANG;
    PUT("CMD",serverConn,&cmd,1,ntimeout);
    PUT("fileNameLenght",serverConn,&fileNameSize,sizeof(uint32_t),ntimeout);
    PUT("fileName",serverConn,fileName.c_str(),fileNameSize,ntimeout);
    GET("nbKangaroo",serverConn,&nbKangaroo,sizeof(uint64_t),ntimeout);
    if(nbRead==0) {
      ::printf("\nFailed to get %s from server\n",fileName.c_str());
      return false;
    }
    if(nbKangaroo==0) {
      return true;
    }

    uint64_t point = (nbKangaroo / KANG_PER_BLOCK) / 32;
    uint64_t pointPrint = 0;

    KBuff = (int256_t*)malloc(KANG_PER_BLOCK * sizeof(int256_t));
    kangs.reserve(nbKangaroo);

    checkSum.SetInt32(0);
    while(nbKangaroo > 0) {
      pointPrint++;
      if(pointPrint > point) {
        ::printf(".");
        pointPrint = 0;
      }
      
      if(nbKangaroo > KANG_PER_BLOCK) {
        nbK = KANG_PER_BLOCK;
      } else {
        nbK = (uint32_t)nbKangaroo;
      }
      
      GETFREE("packet",serverConn,KBuff,nbK * 16,ntimeout,KBuff);

      for(uint32_t k = 0; k < nbK; k++) {
        kangs.push_back(KBuff[k]);
        // Checksum
        Int K;
        K.SetInt32(0);
        K.bits64[3] = KBuff[k].i64[3];
        K.bits64[2] = KBuff[k].i64[2];
        K.bits64[1] = KBuff[k].i64[1];
        K.bits64[0] = KBuff[k].i64[0];
        checkSum.Add(&K);
      }
      nbKangaroo -= nbK;
    }

    free(KBuff);
    Int K;
    K.SetInt32(0);
    GET("checksum",serverConn,K.bits64,32,ntimeout);

    if(!K.IsEqual(&checkSum)) {
      ::printf("\nWarning, Kangaroo backup wrong checksum %s\n",fileName.c_str());
      return false;
    }
  }
  return true;
}

// Send Kangaroo to Server
bool Kangaroo::SendKangaroosToServer(std::string& fileName,std::vector<int256_t>& kangs) {
  int nbWrite;
  uint32_t fileNameSize = (uint32_t)fileName.length();
  uint64_t nbKangaroo = kangs.size();
  uint64_t pos;
  uint32_t nbK;
  int256_t *KBuff;
  Int checkSum;

  WaitForServer();

  uint64_t point = (nbKangaroo/KANG_PER_BLOCK) / 16;
  uint64_t pointPrint = 0;

  if(!endOfSearch) {

    char cmd = SERVER_SAVEKANG;

    PUT("CMD",serverConn,&cmd,1,ntimeout);
    PUT("fileNameLenght",serverConn,&fileNameSize,sizeof(uint32_t),ntimeout);
    PUT("fileName",serverConn,fileName.c_str(),fileNameSize,ntimeout);
    PUT("nbKangaroo",serverConn,&nbKangaroo,sizeof(uint64_t),ntimeout);

    KBuff = (int256_t*)malloc(KANG_PER_BLOCK * sizeof(int256_t));

    checkSum.SetInt32(0);
    pos = 0;
    while(nbKangaroo > 0) {

      pointPrint++;
      if(pointPrint > point) {
        ::printf(".");
        pointPrint = 0;
      }

      if(nbKangaroo> KANG_PER_BLOCK) {
        nbK = KANG_PER_BLOCK;
      } else {
        nbK = (uint32_t)nbKangaroo;
      }

      for(uint32_t k = 0; k < nbK; k++) {
        memcpy(&KBuff[k],&kangs[pos],16);
        pos++;
        // Checksum
        Int K;
        K.SetInt32(0);
        K.bits64[3] = KBuff[k].i64[3];
        K.bits64[2] = KBuff[k].i64[2];
        K.bits64[1] = KBuff[k].i64[1];
        K.bits64[0] = KBuff[k].i64[0];
        checkSum.Add(&K);
      }

      PUTFREE("packet",serverConn,KBuff,nbK * 16,ntimeout,KBuff);

      nbKangaroo -= nbK;
    }

    free(KBuff);

    PUT("checksum",serverConn,checkSum.bits64,32,ntimeout);

  }
  return true;
}

// Send DP to Server
bool Kangaroo::SendToServer(std::vector<ITEM> &dps,uint32_t threadId,uint32_t gpuId) {
  int nbRead;
  int nbWrite;
  uint32_t nbDP = (uint32_t)dps.size();
  if(dps.size()==0)
    return false;

  WaitForServer();
  
  if(!endOfSearch) {
    int32_t status;
    // Send DP
    DP *dp = (DP *)malloc(sizeof(DP)*nbDP);
    for(uint32_t i = 0; i<nbDP; i++) {
      int256_t X;
      int256_t D;
      uint64_t h;
      HashTable::Convert(&dps[i].x,&dps[i].d,dps[i].kIdx % 2,&h,&X,&D);

      dp[i].kIdx = (uint32_t)dps[i].kIdx;
      dp[i].h = (uint32_t)h;
      dp[i].x.i64[0] = X.i64[0];
      dp[i].x.i64[1] = X.i64[1];
      dp[i].x.i64[2] = X.i64[2];
      dp[i].x.i64[3] = X.i64[3];
      dp[i].d.i64[0] = D.i64[0];
      dp[i].d.i64[1] = D.i64[1];
      dp[i].d.i64[2] = D.i64[2];
      dp[i].d.i64[3] = D.i64[3];
    }
    char cmd = SERVER_SENDDP;
    DPHEADER head;
    head.header = SERVER_HEADER;
    head.nbDP = nbDP;
    head.processId = pid;
    head.threadId = threadId;

    PUTFREE("CMD",serverConn,&cmd,1,ntimeout,dp);
    PUTFREE("DPHeader",serverConn,&head,sizeof(DPHEADER),ntimeout,dp);
    PUTFREE("DP",serverConn,dp,sizeof(DP)*nbDP,ntimeout,dp);
    GETFREE("Status",serverConn,&status,sizeof(uint32_t),ntimeout,dp)
    dps.clear();
    free(dp);
  }
  return true;
}

void Kangaroo::AddConnectedClient() {
  connectedClient++;
}

void Kangaroo::RemoveConnectedClient() {
  connectedClient--;
}

void Kangaroo::RemoveConnectedKangaroo(uint64_t nb) {
  totalRW -= nb;
}

// Get configuration from server
bool Kangaroo::GetConfigFromServer() {
  int nbRead;
  int nbWrite;
  
  if(!ConnectToServer(&serverConn)) {
    ::printf("Cannot connect to server: %s\n%s\n",serverIp.c_str(),lastError.c_str());
    return false;
  }

  isConnected = true;
  serverStatus = "Connected";
  Point key;
  key.Clear();
  key.z.SetInt32(1);
  rangeStart.SetInt32(0);
  rangeEnd.SetInt32(0);
  initDPSize = -1;

  // If in pool mode, send the Bitcoin address first
  if (poolMode) {
    // Get the Bitcoin address from command line arguments
    if (bitcoinAddress.empty()) {
      ::printf("Error: Bitcoin address required for pool mode\n");
      close_socket(serverConn);
      isConnected = false;
      return false;
    }
    
    if (!IsValidBitcoinAddress(bitcoinAddress)) {
      ::printf("Error: Invalid Bitcoin address format: %s\n", bitcoinAddress.c_str());
      close_socket(serverConn);
      isConnected = false;
      return false;
    }
    
    uint32_t addrLength = bitcoinAddress.length();
    PUT("AddressLength", serverConn, &addrLength, sizeof(uint32_t), ntimeout);
    PUT("BitcoinAddress", serverConn, bitcoinAddress.c_str(), addrLength, ntimeout);
    
    ::printf("Connected to pool server with Bitcoin address: %s\n", bitcoinAddress.c_str());
  }

  char cmd = SERVER_GETCONFIG;
  PUT("CMD",serverConn,&cmd,1,ntimeout);
  uint32_t version;

  GET("Version",serverConn,&version,sizeof(uint32_t),ntimeout);
  GET("RangeStart",serverConn,rangeStart.bits64,32,ntimeout);
  GET("RangeEnd",serverConn,rangeEnd.bits64,32,ntimeout);
  GET("KeyX",serverConn,key.x.bits64,32,ntimeout);
  GET("KeyY",serverConn,key.y.bits64,32,ntimeout);
  GET("DP",serverConn,&initDPSize,sizeof(int32_t),ntimeout);

  // Set kangaroo number
  cmd = SERVER_SETKNB;
  PUT("CMD",serverConn,&cmd,1,ntimeout);
  PUT("nbKangaroo",serverConn,&totalRW,sizeof(uint64_t),ntimeout);

  ::printf("Succesfully connected to server: %s\n",serverIp.c_str());

  keysToSearch.clear();
  keysToSearch.push_back(key);
  return true;
}
