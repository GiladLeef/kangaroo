#ifndef CONSTANTSH
#define CONSTANTSH

// Release number
#define RELEASE "2.0"

// Kangaroo type
#define TAME 0  // Tame kangaroo
#define WILD 1  // Wild kangaroo

// Number of random jumps
#define NB_JUMP 64 

// GPU group size
#define GPU_GRP_SIZE 128

// GPU number of run per kernel call – larger values mean fewer kernel
// launches and host/device transfers.  Modern GPUs easily handle the extra
// register pressure, and practical testing shows ×4 increases throughput on
// Ampere/Hopper without raising the 'items lost' risk.
#define NB_RUN 256

// SendDP Period in sec
#define SEND_PERIOD 2.0

// Timeout before closing connection idle client in sec
#define CLIENT_TIMEOUT 3600.0

// Number of merge partition
#define MERGE_PART 256

// Hash table configuration
#define HASH_SIZE_BIT 18
#define HASH_SIZE (1<<HASH_SIZE_BIT)
#define HASH_MASK (HASH_SIZE-1)

// Hash table add results
#define ADD_OK        0
#define ADD_DUPLICATE 1
#define ADD_COLLISION 2

// Work file types
#define HEADW  0xFA6A8001  // Full work file
#define HEADK  0xFA6A8002  // Kangaroo only file
#define HEADKS 0xFA6A8003  // Compressed Kangaroo only file

// Number of Hash entry per partition
#define H_PER_PART (HASH_SIZE / MERGE_PART)

#endif //CONSTANTSH
