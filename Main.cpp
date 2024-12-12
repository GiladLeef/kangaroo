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

    auto secp = make_unique<Secp256K1>();
    secp->Init();

    int nbCPUThread = Timer::getCoreNumber();
    string configFile, workFile, iWorkFile, checkWorkFile, infoFile, merge1, merge2, mergeDest, mergeDir, outputFile, serverIP;
    bool saveKangaroo = false, saveKangarooByServer = false, splitWorkFile = false, serverMode = false, checkFlag = false;
    double maxStep = 0.0;
    int dp = -1, wtimeout = DEFAULT_TIMEOUT, ntimeout = DEFAULT_TIMEOUT, port = DEFAULT_PORT;
    uint32_t savePeriod = 60;

    for (int a = 1; a < argc; ++a) {
        string arg = argv[a];
        if (arg == "-v") {
            exit(0);
        } else if (arg == "-t") {
            if (a + 1 < argc) nbCPUThread = getInt("-t", argv[++a]);
        } else if (arg == "-d") {
            if (a + 1 < argc) dp = getInt("-d", argv[++a]);
        } else if (arg == "-w") {
            if (a + 1 < argc) workFile = argv[++a];
        } else if (arg == "-i") {
            if (a + 1 < argc) iWorkFile = argv[++a];
        } else if (arg == "-o") {
            if (a + 1 < argc) outputFile = argv[++a];
        } else if (arg == "-m") {
            if (a + 1 < argc) maxStep = getDouble("-m", argv[++a]);
        } else if (arg == "-s") {
            serverMode = true;
        } else if (arg == "-c") {
            if (a + 1 < argc) serverIP = argv[++a];
        } else if (arg == "-sp") {
            if (a + 1 < argc) port = getInt("-sp", argv[++a]);
        } else if (arg == "-wi") {
            if (a + 1 < argc) savePeriod = getInt("-wi", argv[++a]);
        } else if (arg == "-wt") {
            if (a + 1 < argc) wtimeout = getInt("-wt", argv[++a]);
        } else if (arg == "-nt") {
            if (a + 1 < argc) ntimeout = getInt("-nt", argv[++a]);
        } else if (arg == "-ws") {
            saveKangaroo = true;
        } else if (arg == "-wss") {
            saveKangarooByServer = true;
        } else if (arg == "-wsplit") {
            splitWorkFile = true;
        } else if (arg == "-check") {
            checkFlag = true;
        } else if (a == argc - 1) {
            configFile = argv[a];
        } else {
            cerr << "Unexpected " << argv[a] << " argument" << endl;
            exit(-1);
        }
    }

    auto v = make_unique<Kangaroo>(secp.get(), dp, workFile, iWorkFile, savePeriod, saveKangaroo, saveKangarooByServer,
                                   maxStep, wtimeout, port, ntimeout, serverIP, outputFile, splitWorkFile);

    if (checkFlag) {
        v->Check();
        exit(0);
    }

    if (!checkWorkFile.empty()) {
        v->CheckWorkFile(nbCPUThread, checkWorkFile);
        exit(0);
    }

    if (!infoFile.empty()) {
        v->WorkInfo(infoFile);
        exit(0);
    }

    if (!mergeDir.empty()) {
        v->MergeDir(mergeDir, mergeDest);
        exit(0);
    }

    if (!merge1.empty()) {
        v->MergeWork(merge1, merge2, mergeDest);
        exit(0);
    }

    if (!iWorkFile.empty() && !v->LoadWork(iWorkFile)) {
        exit(-1);
    }

    if (!configFile.empty() && !v->ParseConfigFile(configFile)) {
        exit(-1);
    }

    if (serverMode) {
        v->RunServer();
    } else {
        v->Run(nbCPUThread);
    }

    return 0;
}
