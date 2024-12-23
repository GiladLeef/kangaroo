#include "Kangaroo.h"
#include "Timer.h"
#include "SECPK1/SECP256k1.h"
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <memory>

using namespace std;

constexpr int DEFAULT_TIMEOUT = 3000;
constexpr int DEFAULT_PORT = 17403;

void printUsage() {
    cout << "Kangaroo [-v] [-t nbThread] [-d dpBit] [-check]\n"
         << "         inFile\n"
         << "Options:\n"
         << " -v: Print version\n"
         << " -d: Specify number of leading zeros for the DP method (default is auto)\n"
         << " -t nbThread: Specify number of threads\n"
         << " -w workfile: Specify file to save work into (current processed key only)\n"
         << " -i workfile: Specify file to load work from (current processed key only)\n"
         << " -wi workInterval: Periodic interval (in seconds) for saving work\n"
         << " -ws: Save kangaroos in the work file\n"
         << " -wss: Save kangaroos via the server\n"
         << " -wsplit: Split work file of server and reset hashtable\n"
         << " -wm file1 file2 destfile: Merge work files\n"
         << " -wmdir dir destfile: Merge directory of work files\n"
         << " -wt timeout: Save work timeout in milliseconds (default is 3000ms)\n"
         << " -winfo file1: Work file info file\n"
         << " -wpartcreate name: Create empty partitioned work file (name is a directory)\n"
         << " -wcheck workfile: Check workfile integrity\n"
         << " -m maxStep: Number of operations before giving up the search (maxStep*expected operation)\n"
         << " -s: Start in server mode\n"
         << " -c server_ip: Start in client mode and connect to server server_ip\n"
         << " -sp port: Server port, default is 17403\n"
         << " -nt timeout: Network timeout in milliseconds (default is 3000ms)\n"
         << " -o fileName: Output result to fileName\n"
         << " inFile: Input configuration file\n";
    exit(0);
}

int getInt(const string& name, const string& v) {
    try {
        return stoi(v);
    } catch (const invalid_argument&) {
        cerr << "Invalid " << name << " argument, number expected" << endl;
        exit(-1);
    }
}

double getDouble(const string& name, const string& v) {
    try {
        return stod(v);
    } catch (const invalid_argument&) {
        cerr << "Invalid " << name << " argument, number expected" << endl;
        exit(-1);
    }
}

void getInts(const string& name, vector<int>& tokens, const string& text, char sep) {
    stringstream ss(text);
    string item;
    while (getline(ss, item, sep)) {
        try {
            tokens.push_back(stoi(item));
        } catch (const invalid_argument&) {
            cerr << "Invalid " << name << " argument, number expected" << endl;
            exit(-1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc == 1) {  // No arguments passed
        printUsage();
    }

    cout << "Kangaroo v" RELEASE << endl;

    Timer::Init();
    rseed(Timer::getSeed32());

    // Init SECP256K1
    unique_ptr<Secp256K1> secp = make_unique<Secp256K1>();
    secp->Init();

    int nbCPUThread = Timer::getCoreNumber();
    vector<int> gpuId = {0};
    vector<int> gridSize;
    string workFile, iWorkFile, checkWorkFile, merge1, merge2, mergeDest, mergeDir, infoFile, outputFile;
    bool gpuEnable = false, saveKangaroo = false, saveKangarooByServer = false, splitWorkFile = false, serverMode = false, checkFlag = false;
    string serverIP;
    double maxStep = 0.0;
    int dp = -1, wtimeout = DEFAULT_TIMEOUT, ntimeout = DEFAULT_TIMEOUT, port = DEFAULT_PORT;
    string configFile;
    uint32_t savePeriod = 60;
    
    // Parse arguments
    for (int a = 1; a < argc; ++a) {
        string arg = argv[a];
        if (arg == "-t") {
            if (a + 1 >= argc) {
                cerr << "-t missing argument" << endl;
                exit(0);
            }
            nbCPUThread = getInt("nbCPUThread", argv[++a]);
        } else if (arg == "-d") {
            if (a + 1 >= argc) {
                cerr << "-d missing argument" << endl;
                exit(0);
            }
            dp = getInt("dpSize", argv[++a]);
        } else if (arg == "-w") {
            if (a + 1 >= argc) {
                cerr << "-w missing argument" << endl;
                exit(0);
            }
            workFile = argv[++a];
        } else if (arg == "-i") {
            if (a + 1 >= argc) {
                cerr << "-i missing argument" << endl;
                exit(0);
            }
            iWorkFile = argv[++a];
        } else if (arg == "-gpu") {
            gpuEnable = true;
        } else if (arg == "-gpuId") {
            if (a + 1 >= argc) {
                cerr << "-gpuId missing argument" << endl;
                exit(0);
            }
            getInts("gpuId", gpuId, argv[++a], ',');
        } else if (arg == "-g") {
            if (a + 1 >= argc) {
                cerr << "-g missing argument" << endl;
                exit(0);
            }
            getInts("gridSize", gridSize, argv[++a], ',');
        } else if (arg == "-o") {
            if (a + 1 >= argc) {
                cerr << "-o missing argument" << endl;
                exit(0);
            }
            outputFile = argv[++a];
        } else if (arg == "-s") {
            serverMode = true;
        } else if (arg == "-c") {
            if (a + 1 >= argc) {
                cerr << "-c missing argument" << endl;
                exit(0);
            }
            serverIP = argv[++a];
        } else if (arg == "-sp") {
            if (a + 1 >= argc) {
                cerr << "-sp missing argument" << endl;
                exit(0);
            }
            port = getInt("serverPort", argv[++a]);
        } else if (arg == "-v") {
            exit(0);
        } else if (arg == "-check") {
            checkFlag = true;
        } else {
            configFile = arg;
        }
    }

    if (gridSize.empty()) {
        for (size_t i = 0; i < gpuId.size(); ++i) {
            gridSize.push_back(0);
            gridSize.push_back(0);
        }
    } else if (gridSize.size() != gpuId.size() * 2) {
        cerr << "Invalid gridSize or gpuId argument, must have coherent size" << endl;
        exit(-1);
    }

    unique_ptr<Kangaroo> v = make_unique<Kangaroo>(
        secp.get(), dp, gpuEnable, workFile, iWorkFile, savePeriod, saveKangaroo, saveKangarooByServer, 
        maxStep, wtimeout, port, ntimeout, serverIP, outputFile, splitWorkFile
    );

    if (checkFlag) {
        v->Check();  
        exit(0);
    } else if (!checkWorkFile.empty()) {
        v->CheckWorkFile(nbCPUThread, checkWorkFile);
        exit(0);
    } else if (!infoFile.empty()) {
        v->WorkInfo(infoFile);
        exit(0);
    } else if (!mergeDir.empty()) {
        v->MergeDir(mergeDir, mergeDest);
        exit(0);
    } else if (!merge1.empty()) {
        v->MergeWork(merge1, merge2, mergeDest);
        exit(0);
    } else if (!iWorkFile.empty() && !v->LoadWork(iWorkFile)) {
        exit(-1);
    } else if (!configFile.empty() && !v->ParseConfigFile(configFile)) {
        exit(-1);
    } else if (serverMode) {
        v->RunServer();
    } else {
        v->Run(nbCPUThread, gpuId, gridSize);
    }

    return 0;
}
