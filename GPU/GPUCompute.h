// CUDA Kernel main function

__device__ void ComputeKangaroos(uint64_t *kangaroos,uint32_t maxFound,uint32_t *out,uint64_t dpMask) {

  uint64_t px[GPU_GRP_SIZE][4];
  uint64_t py[GPU_GRP_SIZE][4];
  uint64_t dist[GPU_GRP_SIZE][4];
  uint64_t dx[GPU_GRP_SIZE][4];
  uint64_t dy[4];
  uint64_t rx[4];
  uint64_t ry[4];
  uint64_t altRx[GPU_GRP_SIZE][4];  // Pre-allocate arrays for backward points
  uint64_t altRy[GPU_GRP_SIZE][4];
  uint64_t backDist[GPU_GRP_SIZE][4];
  uint64_t _s[4];
  uint64_t _p[4];
  uint64_t origPx[GPU_GRP_SIZE][4];
  uint64_t origPy[GPU_GRP_SIZE][4];
  uint64_t origDist[GPU_GRP_SIZE][4];
  uint32_t jmp[GPU_GRP_SIZE];

  LoadKangaroos(kangaroos,px,py,dist);
  for(int run = 0; run < NB_RUN; run++) {

    // P1 = jumpPoint
    // P2 = kangaroo
    
    __syncthreads();
    
    // First pass - prepare data and calculate dx
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      jmp[g] = (uint32_t)px[g][0] & (NB_JUMP-1);
      ModSub256(dx[g],px[g],jPx[jmp[g]]);
      
      // Save original point and distance
      Load256(origPx[g],px[g]);
      Load256(origPy[g],py[g]);
      Load256(origDist[g],dist[g]);
    }
    
    // Batch inversion - most expensive operation
    _ModInvGrouped(dx);
    __syncthreads();

    // Second pass - calculate both forward and backward points
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      ModSub256(dy,py[g],jPy[jmp[g]]);
      _ModMult(_s,dy,dx[g]);
      _ModSqr(_p,_s);
      
      // Forward jump point calculation
      ModSub256(rx,_p,jPx[jmp[g]]);
      ModSub256(rx,px[g]);
      ModSub256(ry,px[g],rx);
      _ModMult(ry,_s);
      ModSub256(ry,py[g]);
      
      // Backward jump point calculation (cheap second-point trick)
      ModSub256(altRx[g],_p,jPx[jmp[g]]);
      ModSub256(altRx[g],px[g]);
      ModNeg256(altRx[g]);
      
      ModSub256(altRy[g],px[g],altRx[g]);
      _ModMult(altRy[g],_s);
      ModSub256(altRy[g],py[g]);
      ModNeg256(altRy[g]);
      
      // Prepare backward distance 
      Load256(backDist[g],origDist[g]);
      ModSub256(backDist[g],backDist[g],jD[jmp[g]]);
      
      // Store forward point
      Load256(px[g],rx);
      Load256(py[g],ry);
      Add256(dist[g],jD[jmp[g]]);
    }
    
    // Third pass - check for distinguished points in batch
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      // Check if forward point is a distinguished point
      if((px[g][3] & dpMask) == 0) {
        // Distinguished point
        uint32_t pos = atomicAdd(out,1);
        if(pos < maxFound) {
          uint64_t kIdx = (uint64_t)IDX + (uint64_t)g*(uint64_t)blockDim.x + (uint64_t)blockIdx.x*((uint64_t)blockDim.x*GPU_GRP_SIZE);
          OutputDP(px[g],dist[g],&kIdx);
        }
      }
      
      // Check if backward point is a distinguished point
      if((altRx[g][3] & dpMask) == 0) {
        // Distinguished point
        uint32_t pos = atomicAdd(out,1);
        if(pos < maxFound) {
          uint64_t kIdx = (uint64_t)IDX + (uint64_t)g*(uint64_t)blockDim.x + (uint64_t)blockIdx.x*((uint64_t)blockDim.x*GPU_GRP_SIZE);
          OutputDP(altRx[g],backDist[g],&kIdx);
        }
      }
    }
  }
  StoreKangaroos(kangaroos,px,py,dist);
}
