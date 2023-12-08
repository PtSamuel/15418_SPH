#include "Particle.h"
#include "bitonic_sort.h"

#include <stdio.h>

#define BLOCK_DIM 16
// DO NOT DO THIS!!!
// #define THREADS_PER_BLOCK BLOCK_DIM * BLOCK_DIM
#define THREADS_PER_BLOCK (BLOCK_DIM * BLOCK_DIM)

__device__ void cas(Particle *p1, Particle *p2, int polarity) {
    bool misordered = p1->block > p2->block || (p1->block == p2->block && p1->id > p2->id);
    if(!polarity) {
        if(misordered) {
            Particle temp = *p1;
            *p1 = *p2;
            *p2 = temp;
        }
    } else {
        if(!misordered) {
            Particle temp = *p1;
            *p1 = *p2;
            *p2 = temp;
        }
    }
}

__global__ void compare_and_swap(Particle *particles, int n, int stride, int groupsize) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;

    if(index >= n / 2) return;

    int groupid = index / groupsize;
    int groupstart = groupsize * 2 * groupid;
    int groupsubid = index - groupid * groupsize;
    int groupmatesubid = groupsubid + groupsize;

    int sort_order = (groupstart / stride) % 2;

    printf("thread index: %d, groupstart: %d, subid: %d, matesubid: %d, order: %d\n", index, groupstart, groupsubid, groupmatesubid, sort_order);

    // particles[groupstart + groupsubid].block = 0;
    // particles[groupstart + groupmatesubid].block = 0;

    cas(&particles[groupstart + groupsubid], &particles[groupstart + groupmatesubid], sort_order);
}

void bitonic_sort(Particle *p, int n) {
    Particle *particles;
    cudaMalloc(&particles, sizeof(Particle) * n);
    cudaMemcpy(particles, p, sizeof(Particle) * n, cudaMemcpyHostToDevice);

    int num_tasks = n / 2;

    int num_blocks = (num_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    
    int stage = 1;
    for(int stride = 2; stride <= n; stride *= 2) {
        for(int groupsize = stage; groupsize >= 1; groupsize /= 2) {
            printf("stride: %d, groupsize: %d\n", stride, groupsize);
            compare_and_swap<<<grid_dim, block_dim>>>(particles, n, stride, groupsize);
            cudaDeviceSynchronize();
        }
        stage *= 2;
    }

    cudaMemcpy(p, particles, sizeof(Particle) * n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) {
        printf("%d: %d\n", p[i].id, p[i].block);
    }
}