## Pollard's Kangaroo for SECP256K1

The source code of this software can be found at https://github.com/giladleef/kangaroo.

This program offers a powerful solution for tackling the Elliptic Curve Discrete Logarithm Problem (ECDLP) within the context of SECP256K1. Here's a detailed overview of the features and functionality of this solver:
### Features

- **256-bit, Fixed-size arithmetic:** Utilizes 4*64 bits, for fixed-size arithmetic for efficient computation.
- **254-bit search range:** The search range extended to 254 bits. (2 bits are used for flags).
- **Endomorphism optimization:** Comparing only the Y-coordinates enables finding a collision three times faster.
- **Fast modular inversion:** Implements fast modular inversion using Delayed Right Shift 62 bits.
- **Fast modular multiplication:** Utilizes 2 steps folding 512 bits to 256 bits reduction using 64-bit digits.
- **Efficient HashTable Implementation:** State-of-the-art HashTable implementation.
- **Multi-GPU support:** with CUDA optimisation via inline PTX assembly.
- **Reduced code cycles:** Compared with the original version, this version reduces redundant code cycles.
- **Native windows support:** The CPU version can be compiled directly with both gcc/clang on a windows machine.

This program is based on https://github.com/JeanLucPons/Kangaroo.
### How It Works

The algorithm employs two herds of kangaroos, a tame herd, and a wild herd. When a kangaroo from each herd collides, the key can be solved. The distinguished points method with a hashtable is used to detect collisions efficiently. The algorithm iteratively updates the positions of the kangaroos until a collision is detected, leading to the solution of the ECDLP.

### Input File Structure

The input file follows a specific structure:

```
Start range
End range
Key #1
Key #2
...
```

For example:

```
0
FFFFFFFFFFFFFF
02E9F43F810784FF1E91D8BC7C4FF06BFEE935DA71D7350734C3472FE305FEF82A
```

### Probability of Success

The probability of success after a certain number of group operations is illustrated, considering the range size (N). The plot provides insights into the likelihood of solving the ECDLP within a given range.
![successprob](https://github.com/GiladLeef/kangaroo/assets/96906027/bd7865f5-1eef-4207-b6a6-eac80a5064bb)


### Time/Memory Tradeoff of the DP Method

The distinguished point (DP) method offers an efficient approach for storing random walks and detecting collisions between them. It stores only points with an x value starting with a specified number of zero bits. However, there's a tradeoff when dealing with a large number of kangaroos and a small range, as it may lead to increased overhead and memory usage. Adjusting the DP mask size can help optimize performance.

### Dealing with Work Files

Work files can be saved periodically using various options (-w, -wi, -ws). When restarting a work, the -i option can be used, and work files can be merged offline. Work files are compatible and can be merged if they have the same key and range. The -wss option enables using the server to make kangaroo backups, facilitating work continuity across different configurations or hardware setups.


### Compile on Ubuntu 24.04 x86_64 Machine

```
# Install a CUDA driver
sudo apt-get install -y nvidia-open

# Install Nvidia CUDA Toolkit

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Install g++ and make
sudo apt-get install make g++

# Compile CPU-ONLY version
make

# Compile with GPU support:
make gpu=1 ccap=<integer> all
