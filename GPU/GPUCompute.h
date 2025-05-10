// CUDA Kernel main function

__device__ void ComputeKangaroos(uint64_t *kangaroos,uint32_t maxFound,uint32_t *out,uint64_t dpMask) {

  uint64_t px[GPU_GRP_SIZE][4];
  uint64_t py[GPU_GRP_SIZE][4];
  uint64_t dist[GPU_GRP_SIZE][4];
  uint64_t dx[GPU_GRP_SIZE][4];
  uint64_t dy[4];
  uint64_t rx[4];
  uint64_t ry[4];
  uint64_t altRx[4];  // For backward jump
  uint64_t altRy[4];  // For backward jump
  uint64_t _s[4];
  uint64_t _p[4];
  uint64_t origPx[4];
  uint64_t origPy[4];
  uint64_t origDist[4];
  uint32_t jmp;

  LoadKangaroos(kangaroos,px,py,dist);
  for(int run = 0; run < NB_RUN; run++) {

    // P1 = jumpPoint
    // P2 = kangaroo
    
    __syncthreads();
    
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      jmp = (uint32_t)px[g][0] & (NB_JUMP-1);
      ModSub256(dx[g],px[g],jPx[jmp]);
    }
    _ModInvGrouped(dx);
    __syncthreads();

    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      jmp = (uint32_t)px[g][0] & (NB_JUMP-1);
      
      // Save original point and distance
      Load256(origPx,px[g]);
      Load256(origPy,py[g]);
      Load256(origDist,dist[g]);
      
      ModSub256(dy,py[g],jPy[jmp]);
      _ModMult(_s,dy,dx[g]);
      _ModSqr(_p,_s);
      
      // Forward jump point calculation
      ModSub256(rx,_p,jPx[jmp]);
      ModSub256(rx,px[g]);
      ModSub256(ry,px[g],rx);
      _ModMult(ry,_s);
      ModSub256(ry,py[g]);
      
      // Backward jump point calculation (cheap second-point trick)
      ModSub256(altRx,_p,jPx[jmp]);
      ModSub256(altRx,px[g]);
      ModNeg256(altRx);
      
      ModSub256(altRy,px[g],altRx);
      _ModMult(altRy,_s);
      ModSub256(altRy,py[g]);
      ModNeg256(altRy);
      
      // Store forward point
      Load256(px[g],rx);
      Load256(py[g],ry);
      Add256(dist[g],jD[jmp]);
      
      // Check if forward point is a distinguished point
      if((px[g][3] & dpMask) == 0) {
        // Distinguished point
        uint32_t pos = atomicAdd(out,1);
        if(pos < maxFound) {
          uint64_t kIdx = (uint64_t)IDX + (uint64_t)g*(uint64_t)blockDim.x + (uint64_t)blockIdx.x*((uint64_t)blockDim.x*GPU_GRP_SIZE);
          OutputDP(px[g],dist[g],&kIdx);
        }
      }
      
      // Prepare backward distance by subtracting jump distance from original
      uint64_t backDist[4];
      Load256(backDist,origDist);
      Sub256(backDist,jD[jmp]);
      
      // Check if backward point is a distinguished point
      if((altRx[3] & dpMask) == 0) {
        // Distinguished point
        uint32_t pos = atomicAdd(out,1);
        if(pos < maxFound) {
          uint64_t kIdx = (uint64_t)IDX + (uint64_t)g*(uint64_t)blockDim.x + (uint64_t)blockIdx.x*((uint64_t)blockDim.x*GPU_GRP_SIZE);
          OutputDP(altRx,backDist,&kIdx);
        }
      }
    }
  }
  StoreKangaroos(kangaroos,px,py,dist);
}
