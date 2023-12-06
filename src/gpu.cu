#include <string>
#include <math.h>
#include <cuda_runtime.h>

#include "gpu.h"
#include "Particle.h"

#define BLOCK_DIM 16
#define THREADS_PER_BLOCK BLOCK_DIM * BLOCK_DIM

#define SMOOTH_RADIUS 1.0f
#define SMOOTH_RADIUS2 SMOOTH_RADIUS * SMOOTH_RADIUS
#define SMOOTH_RADIUS4 SMOOTH_RADIUS2 * SMOOTH_RADIUS2

static const float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
static const float normalizer = 1 / kernel_volume;

static float *particles;
static float *densities;

struct CUDAParams {
    float *particles;
    float *densities;
};
__constant__ CUDAParams params;

__device__ float smoothing_kernal(float2 disp) {
    float dist = sqrt(disp.x * disp.x + disp.y * disp.y);
    float offset = fmax(0.0f, SMOOTH_RADIUS - dist);
    return offset * offset * normalizer;
}

__global__ void compute_density(int n) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    printf("%d, %p\n", index, params.particles);
    Particle cur = *(Particle*)&params.particles[index * sizeof(Particle)];
    float2 pos = make_float2(
        cur.pos.x,
        cur.pos.y
    );
    float density = 0;

    for(int i = 0; i < n; i++) {
        Particle p = *(Particle*)&params.particles[i * sizeof(Particle)];

        float2 disp = make_float2(
            pos.x - p.pos.x,
            pos.y - p.pos.y
        );
        density += smoothing_kernal(disp);
    }

    params.densities[index] = density;
}

void show_device() {
    int device_count = 0;

    std::string name;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, i);
        name = device_props.name;

        printf("Device %d: %s\n", i, device_props.name);
        printf("   SMs:        %d\n", device_props.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(device_props.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", device_props.major, device_props.minor);
    }
    
    printf("---------------------------------------------------------\n");   
}

void gpu_init(int n) {
    cudaMalloc(&particles, n * sizeof(Particle));
    cudaMalloc(&densities, n * sizeof(float));
    CUDAParams p;
    printf("init: %p\n", particles);
    p.particles = particles;
    p.densities = densities;

    // It is params, not &params
    // cudaMemcpyToSymbol(&params, &p, sizeof(CUDAParams));

    cudaMemcpyToSymbol(params, &p, sizeof(CUDAParams));
    printf("init: %p\n", params.particles);
}

void compute_densities_gpu(Particle *p, int n, float* dst) {
    cudaMemcpy(particles, p, n * sizeof(Particle), cudaMemcpyHostToDevice);
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    compute_density<<<grid_dim, block_dim>>>(n);
    cudaDeviceSynchronize();

    cudaMemcpy(dst, densities, n * sizeof(float), cudaMemcpyDeviceToHost);
}