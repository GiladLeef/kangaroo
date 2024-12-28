#ifndef CONSTANTSH
#define CONSTANTSH

// Release number
#define RELEASE "2.0"

// Kangaroo type
#define TAME 0  // Tame kangaroo
#define WILD 1  // Wild kangaroo

// Number of random jumps
#define NB_JUMP 32 // Max 512 for the GPU

// GPU group size
#define GPU_GRP_SIZE 128

// GPU number of run per kernel call
#define NB_RUN 64

// SendDP Period in sec
#define SEND_PERIOD 2.0

// Timeout before closing connection idle client in sec
#define CLIENT_TIMEOUT 3600.0

// Number of merge partition
#define MERGE_PART 256

#endif //CONSTANTSH
