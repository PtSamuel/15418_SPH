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

#define PRESSURE_RESPONSE 200.0f

static const float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
static const float normalizer = 1 / kernel_volume;

static uchar1 *particles;
static float *densities;
static float *pressures;

struct CUDAParams {
    uchar1 *particles;
    float *densities;
    float *pressures;
    float desired_density;
};
__constant__ CUDAParams params;

__device__ float smoothing_kernal(float2 disp) {
    float dist = sqrt(disp.x * disp.x + disp.y * disp.y);
    float offset = fmax(0.0f, SMOOTH_RADIUS - dist);
    return offset * offset * normalizer;
}

__device__ __inline__ void print_particle(Particle &p) {
    printf("pos %d: (%f, %f), vel: (%f, %f)\n", p.id, p.pos.x, p.pos.y, p.vel.x, p.vel.y);
}

// __global__ void compute_density_and_pressure(int n) {
//     int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
//     if(index >= n) return;

    
//     Particle cur = *(Particle*)&params.particles[index * sizeof(Particle)];
//     // print_particle(cur);
//     float2 pos = make_float2(
//         cur.pos.x,
//         cur.pos.y
//     );
//     float density = 0;

//     for(int i = 0; i < n; i++) {
//         Particle p = *(Particle*)&params.particles[i * sizeof(Particle)];

//         float2 disp = make_float2(
//             pos.x - p.pos.x,
//             pos.y - p.pos.y
//         );
//         density += smoothing_kernal(disp);
//     }

//     // printf("%d: %f, %d\n", index, density, cur.id);

//     params.densities[index] = density;
    
//     // float pressure = PRESSURE_RESPONSE * (density - params.desired_density);
//     // params.pressures[index] = pressure;
// }

__global__ void compute_density(int n) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    
    Particle cur = *(Particle*)&params.particles[index * sizeof(Particle)];
    // print_particle(cur);
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

    // printf("%d: %f, %d\n", index, density, cur.id);

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

void gpu_init(int n, float desired_density) {

    cudaMalloc(&particles, n * sizeof(Particle));
    cudaMalloc(&densities, n * sizeof(float));
    cudaMalloc(&pressures, n * sizeof(float));

    CUDAParams p;
    p.particles = particles;
    p.densities = densities;
    p.pressures = pressures;
    p.desired_density = desired_density;

    // It is params, not &params
    // cudaMemcpyToSymbol(&params, &p, sizeof(CUDAParams));

    cudaMemcpyToSymbol(params, &p, sizeof(CUDAParams));
}

void compute_densities_and_pressures_gpu(Particle *p, int n, float* dst_density, float *dst_pressure) {
    cudaMemcpy(particles, p, n * sizeof(Particle), cudaMemcpyHostToDevice);
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    // compute_density_and_pressure<<<grid_dim, block_dim>>>(n);
    compute_density<<<grid_dim, block_dim>>>(n);

    cudaDeviceSynchronize();

    cudaMemcpy(dst_density, densities, n * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(dst_pressure, pressures, n * sizeof(float), cudaMemcpyDeviceToHost);
}

// void compute_pressures_gpu(float *p, int n, float* dst) {
//     cudaMemcpy(prssures, p, n * sizeof(float), cudaMemcpyHostToDevice);
//     int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
//     dim3 grid_dim(num_blocks, 1);
//     dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
//     compute_pressures<<<grid_dim, block_dim>>>(n);
//     cudaDeviceSynchronize();

//     cudaMemcpy(dst, pressures, n * sizeof(float), cudaMemcpyDeviceToHost);
// }