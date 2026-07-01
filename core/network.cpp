#include "kangaroo.h"
#include <fstream>
#include "intgroup.h"
#include "timer.h"
#include "workfile.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <signal.h>
#include <unordered_map>
#include <memory>
#include <array>
#include <utility>
#include <vector>
#include <filesystem>
#include <mutex>

using namespace std;
namespace fs = std::filesystem;

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

// Status
#define SERVER_OK            0
#define SERVER_END           1
#define SERVER_BACKUP        2


#define close_socket(s) close(s)

constexpr uint32_t MAX_NETWORK_FILENAME = 256;

namespace {

uint32_t KangarooBlockCount(uint64_t remaining) {
  return remaining > KANG_PER_BLOCK ? KANG_PER_BLOCK : (uint32_t)remaining;
}

void AddKangarooChecksum(Int& checkSum, const int256_t& raw) {
  Int K;
  K.SetInt32(0);
  K.bits64[3] = raw.i64[3];
  K.bits64[2] = raw.i64[2];
  K.bits64[1] = raw.i64[1];
  K.bits64[0] = raw.i64[0];
  checkSum.Add(&K);
}

template <typename ReadBlock, typename WriteBlock>
bool TransferKangarooBlocks(uint64_t& nbKangaroo,std::vector<int256_t>& buffer,Int& checkSum,ReadBlock&& readBlock,WriteBlock&& writeBlock) {
  checkSum.SetInt32(0);
  while(nbKangaroo > 0) {
    uint32_t nbK = KangarooBlockCount(nbKangaroo);
    if(!readBlock(nbK,buffer)) {
      return false;
    }
    for(uint32_t k = 0; k < nbK; k++) {
      AddKangarooChecksum(checkSum, buffer[k]);
    }
    if(!writeBlock(nbK,buffer)) {
      return false;
    }
    nbKangaroo -= nbK;
  }
  return true;
}

}  // namespace

string GetNetworkError() {

  return string(strerror(errno));

}

#define GET(name,s,b,bl,t)  if( (nbRead=Read(s,(char *)(b),bl,t))<0 ) { ::printf("\nReadError(" name "): %s\n",lastError.c_str()); isConnected = false; close_socket(s); return false; }
#define PUT(name,s,b,bl,t)  if( (nbWrite=Write(s,(char *)(b),bl,t))<0 ) { ::printf("\nWriteError(" name "): %s\n",lastError.c_str()); isConnected = false; close_socket(s); return false; }
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
::printf("\nClosing connection with %s\n",p->clientInfo.c_str()); \
close_socket(p->clientSock); \
return false;


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
      ::printf("\nNew connection from %s\n",p->clientInfo.c_str());
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
      std::array<char, 5> response{};
      collisionInSameHerd = 0;
      GET("flush",p->clientSock,response.data(),2,ntimeout);
      std::snprintf(response.data(), response.size(), "OK\n");
      PUT("resp",p->clientSock,response.data(),3,ntimeout);
    } break;

    case SERVER_LOADKANG: {
      Int checkSum;
      uint64_t nbKangaroo = 0;
      uint32_t strSize;
      vector<int256_t> KBuff(KANG_PER_BLOCK);
      uint32_t header = HEADKS;
      uint32_t version = 0;
      std::string fileName;

      GET("fileNameLenght",p->clientSock,&strSize,sizeof(uint32_t),ntimeout);
      if(strSize >= MAX_NETWORK_FILENAME) {
        ::printf("\nFileName too long (MAX=256) %s\n",p->clientInfo.c_str());
        CLIENT_ABORT();
      }

      fileName.resize(strSize);
      GET("fileName",p->clientSock,fileName.data(),strSize,ntimeout);
      workfile::FileHandle f = workfile::openFile(fileName, "rb");
      if(!f) {
        // No backup
        ::printf("LoadKang: Cannot open %s for reading\n",fileName.c_str());
        ::printf("%s\n",::strerror(errno));
        PUT("nbKangaroo",p->clientSock,&nbKangaroo,sizeof(uint64_t),ntimeout);
        break;
      }

      if(!workfile::readValue(f, header)) {
        ::printf("LoadKang: Cannot read from %s\n",fileName.c_str());
        ::printf("%s\n",::strerror(errno));
        CLIENT_ABORT();
      }

      if(header!=HEADKS) {
        ::printf("LoadKang: %s Not a compressed kangaroo file\n",fileName.c_str());
        ::printf("%s\n",::strerror(errno));
        CLIENT_ABORT();
      }

      workfile::readValue(f, version);
      workfile::readValue(f, nbKangaroo);

      PUT("nbKangaroo",p->clientSock,&nbKangaroo,sizeof(uint64_t),ntimeout);

      auto readBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
        for(uint32_t k = 0; k < nbK; k++) {
          if(::fread(&buffer[k],16,1,f.get()) != 1) {
            return false;
          }
        }
        return true;
      };
      auto writeBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
        return Write(p->clientSock,(char*)buffer.data(),(int)(nbK * 16),ntimeout) == (int)(nbK * 16);
      };
      if(!TransferKangarooBlocks(nbKangaroo,KBuff,checkSum,readBlock,writeBlock)) {
        CLIENT_ABORT();
      }

      PUT("checkSum",p->clientSock,checkSum.bits64,32,ntimeout);
    } break;

    case SERVER_SAVEKANG: {
      Int checkSum;
      Int K;
      uint64_t nbKangaroo;
      uint32_t fileNameSize;
      vector<int256_t> KBuff(KANG_PER_BLOCK);
      uint32_t header = HEADKS;
      uint32_t version = 0;
      std::string fileName;

      GET("fileNameLenght",p->clientSock,&fileNameSize,sizeof(uint32_t),ntimeout);
      if(fileNameSize >= MAX_NETWORK_FILENAME) {
        ::printf("\nFileName too long (MAX=256) %s\n",p->clientInfo.c_str());
        CLIENT_ABORT();
      }

      fileName.resize(fileNameSize);
      GET("fileName",p->clientSock,fileName.data(),fileNameSize,ntimeout);
      GET("nbKangaroo",p->clientSock,&nbKangaroo,sizeof(uint64_t),ntimeout);

      std::string fileNameTmp = fileName + ".tmp";

      workfile::FileHandle f = workfile::openFile(fileNameTmp, "wb");
      if(!f) {
        ::printf("\nCannot open %s for writing\n",fileNameTmp.c_str());
        ::printf("%s\n",::strerror(errno));
        CLIENT_ABORT();
      }

      if(!workfile::writeValue(f, header)) {
        ::printf("\nCannot write to %s\n",fileNameTmp.c_str());
        ::printf("%s\n",::strerror(errno));
        CLIENT_ABORT();
      }
      workfile::writeValue(f, version);
      workfile::writeValue(f, nbKangaroo);
      
      auto readBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
        return Read(p->clientSock,(char*)buffer.data(),(int)(nbK * 16),ntimeout) == (int)(nbK * 16);
      };
      auto writeBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
        for(uint32_t k = 0; k < nbK; k++) {
          if(::fwrite(&buffer[k],16,1,f.get()) != 1) {
            return false;
          }
        }
        return true;
      };
      if(!TransferKangarooBlocks(nbKangaroo,KBuff,checkSum,readBlock,writeBlock)) {
        CLIENT_ABORT();
      }

      K.SetInt32(0);
      GET("checksum",p->clientSock,K.bits64,32,ntimeout);

      if(!K.IsEqual(&checkSum)) {
        ::printf("\nWarning, Kangaroo backup wrong checksum %s\n",fileName.c_str());
      } else {
        fs::remove(fileName);
        fs::rename(fileNameTmp,fileName);
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
          ::printf("\nUnexpected DP header from %s\n", p->clientInfo.c_str());
          CLIENT_ABORT();
      }

      if (head.nbDP == 0) {
          ::printf("\nUnexpected number of DP [%d] from %s\n", head.nbDP, p->clientInfo.c_str());
          CLIENT_ABORT();
      } else {
          vector<DP> dp(head.nbDP);
          GET("DP", p->clientSock, dp.data(), sizeof(DP) * head.nbDP, ntimeout);
          state = GetServerStatus();
          PUT("Status", p->clientSock, &state, sizeof(int32_t), ntimeout);
          
          if (nbRead != sizeof(DP) * head.nbDP) {
              ::printf("\nUnexpected DP size from %s [nbDP=%d, Got %d, Expected %d]\n",
                  p->clientInfo.c_str(), head.nbDP, nbRead, (int)(sizeof(DP) * head.nbDP));
              CLIENT_ABORT();
          } else {
              std::lock_guard<std::mutex> lock(ghMutex);
              DP_CACHE dc;
              dc.dp = std::move(dp);
              recvDP.push_back(dc);
          }
      }
    } break;

    default:
      ::printf("\nUnexpected command [%d] from %s\n",cmdBuff,p->clientInfo.c_str());
      CLIENT_ABORT();
    }
  }

  close_socket(p->clientSock);
  return true;
}

void *_acceptThread(void *lpParam) {
  std::unique_ptr<TH_PARAM> p((TH_PARAM *)lpParam);
  p->obj->AddConnectedClient();
  p->obj->HandleRequest(p.get());
  p->obj->RemoveConnectedClient();
  p->obj->RemoveConnectedKangaroo(p->nbKangaroo);
  p->isRunning = false;
  return 0;
}

void *_processServer(void *lpParam) {
  Kangaroo *obj = (Kangaroo *)lpParam;
  obj->ProcessServer();
  return 0;
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
      std::unique_ptr<TH_PARAM> p = std::make_unique<TH_PARAM>();
      p->clientInfo = std::string(inet_ntoa(client_add.sin_addr)) + ":" + std::to_string(ntohs(client_add.sin_port));
      p->obj = this;
      p->isRunning = true;
      p->clientSock = clientSock;
      LaunchThread(_acceptThread,p.get()).detach();
      p.release();
    }
  }
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
  LaunchThread(_processServer,(TH_PARAM *)this).detach();
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
  soc_addr = {};
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
  if(hostInfo.empty()) {

    if(signal(SIGINT,sig_handler) == SIG_ERR)
      ::printf("\nWarning:can't install singal handler\n");

    struct hostent *host_info;
    host_info = gethostbyname(serverIp.c_str());
    if(host_info == NULL) {
      lastError = "Unknown host:" + serverIp;
      hostInfo.clear();
      return false;
    } else {
      hostInfo.assign(host_info->h_addr, host_info->h_addr + host_info->h_length);
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
  server = {};
  server.sin_family = hostAddrType;
  ::memcpy((char*)&server.sin_addr,hostInfo.data(),hostInfo.size());
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
  vector<int256_t> KBuff(KANG_PER_BLOCK);
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

    kangs.reserve(nbKangaroo);

    auto readBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
      for(uint32_t k = 0; k < nbK; k++) {
        if(Read(serverConn,(char*)&buffer[k],16,ntimeout) != 16) {
          return false;
        }
      }
      return true;
    };
    auto writeBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
      for(uint32_t k = 0; k < nbK; k++) {
        kangs.push_back(buffer[k]);
      }
      return true;
    };
    if(!TransferKangarooBlocks(nbKangaroo,KBuff,checkSum,readBlock,writeBlock)) {
      return false;
    }

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
  std::lock_guard<std::mutex> l(ghMutex);
  int nbWrite;
  uint32_t fileNameSize = (uint32_t)fileName.length();
  uint64_t nbKangaroo = kangs.size();
  uint64_t pos;
  vector<int256_t> KBuff(KANG_PER_BLOCK);
  Int checkSum;

  WaitForServer();

  if(!endOfSearch) {

    char cmd = SERVER_SAVEKANG;

    PUT("CMD",serverConn,&cmd,1,ntimeout);
    PUT("fileNameLenght",serverConn,&fileNameSize,sizeof(uint32_t),ntimeout);
    PUT("fileName",serverConn,fileName.c_str(),fileNameSize,ntimeout);
    PUT("nbKangaroo",serverConn,&nbKangaroo,sizeof(uint64_t),ntimeout);

    pos = 0;
    auto readBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
      for(uint32_t k = 0; k < nbK; k++) {
        memcpy(&buffer[k],&kangs[pos],16);
        pos++;
      }
      return true;
    };
    auto writeBlock = [&](uint32_t nbK,std::vector<int256_t>& buffer) {
      return Write(serverConn,(char*)buffer.data(),(int)(nbK * 16),ntimeout) == (int)(nbK * 16);
    };
    if(!TransferKangarooBlocks(nbKangaroo,KBuff,checkSum,readBlock,writeBlock)) {
      return false;
    }

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

  {
    std::lock_guard<std::mutex> l(ghMutex);
    WaitForServer();
  }
  
  if(!endOfSearch) {
    int32_t status;
    // Send DP
    vector<DP> dp(nbDP);
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

    {
      std::lock_guard<std::mutex> l(ghMutex);
      PUT("CMD",serverConn,&cmd,1,ntimeout);
      PUT("DPHeader",serverConn,&head,sizeof(DPHEADER),ntimeout);
      PUT("DP",serverConn,dp.data(),sizeof(DP)*nbDP,ntimeout);
      GET("Status",serverConn,&status,sizeof(uint32_t),ntimeout)
    }
    dps.clear();
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
