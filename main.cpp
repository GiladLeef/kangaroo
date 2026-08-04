#include "kangaroo.h"

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

string nextArgOrDie(int& a, int argc, char* argv[], const string& opt) {
    if (a + 1 >= argc) {
        cerr << opt << " missing argument" << endl;
        exit(0);
    }
    return argv[++a];
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
            nbCPUThread = getInt("nbCPUThread", nextArgOrDie(a, argc, argv, "-t"));
        } else if (arg == "-d") {
            dp = getInt("dpSize", nextArgOrDie(a, argc, argv, "-d"));
        } else if (arg == "-w") {
            workFile = nextArgOrDie(a, argc, argv, "-w");
        } else if (arg == "-i") {
            iWorkFile = nextArgOrDie(a, argc, argv, "-i");
        } else if (arg == "-gpu") {
            gpuEnable = true;
        } else if (arg == "-gpuId") {
            getInts("gpuId", gpuId, nextArgOrDie(a, argc, argv, "-gpuId"), ',');
        } else if (arg == "-g") {
            getInts("gridSize", gridSize, nextArgOrDie(a, argc, argv, "-g"), ',');
        } else if (arg == "-o") {
            outputFile = nextArgOrDie(a, argc, argv, "-o");
        } else if (arg == "-s") {
            serverMode = true;
        } else if (arg == "-c") {
            serverIP = nextArgOrDie(a, argc, argv, "-c");
        } else if (arg == "-sp") {
            port = getInt("serverPort", nextArgOrDie(a, argc, argv, "-sp"));
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
