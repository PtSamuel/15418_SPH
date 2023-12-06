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
static uchar1 *particles_swap;
static float *densities;
static float *pressures;
static float2 *pressure_grads;
static uchar1 *x_dots;

struct StateDerivateCUDA {
    float2 vel;
    float2 acc;
};

struct CUDAParams {
    uchar1 *particles;
    uchar1 *particles_swap;
    float *densities;
    float *pressures;
    float2 *pressure_grads;
    uchar1 *x_dots;

    float desired_density;
};
__constant__ CUDAParams params;

__device__ float smoothing_kernal(float2 disp) {
    float dist = sqrt(disp.x * disp.x + disp.y * disp.y);
    float offset = fmax(0.0f, SMOOTH_RADIUS - dist);
    return offset * offset * normalizer;
}

__device__ float2 smoothing_kernal_grad(float2 disp) {
    float dist2 = disp.x * disp.x + disp.y * disp.y;
    if(dist2 == 0.0f || dist2 > SMOOTH_RADIUS2)
        return make_float2(0.0f, 0.0f);
    
    float dist = sqrt(dist2);
    float x = -2 * (SMOOTH_RADIUS - dist) * disp.x / dist * normalizer;
    float y = -2 * (SMOOTH_RADIUS - dist) * disp.y / dist * normalizer;
    return make_float2(x, y);
}


__device__ __inline__ void print_particle(Particle &p) {
    printf("pos %d: (%f, %f), vel: (%f, %f)\n", p.id, p.pos.x, p.pos.y, p.vel.x, p.vel.y);
}

__global__ void compute_density_and_pressure(int n) {
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
    float pressure = PRESSURE_RESPONSE * (density - params.desired_density);
    params.pressures[index] = pressure;
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

    // printf("size of float2: %ld\n", sizeof(float2));

    cudaMalloc(&particles, n * sizeof(Particle));
    cudaMalloc(&particles_swap, n * sizeof(Particle));
    cudaMalloc(&densities, n * sizeof(float));
    cudaMalloc(&pressures, n * sizeof(float));
    cudaMalloc(&pressure_grads, n * sizeof(float2));
    cudaMalloc(&x_dots, n * sizeof(StateDerivateCUDA));

    CUDAParams p;
    p.particles = particles;
    p.particles_swap = particles_swap;
    p.densities = densities;
    p.pressures = pressures;
    p.pressure_grads = pressure_grads;
    p.desired_density = desired_density;
    p.x_dots = x_dots;

    // It is params, not &params
    // cudaMemcpyToSymbol(&params, &p, sizeof(CUDAParams));

    cudaMemcpyToSymbol(params, &p, sizeof(CUDAParams));
}

void compute_densities_and_pressures_gpu(Particle *p, int n, float* dst_density, float *dst_pressure) {
    cudaMemcpy(particles, p, n * sizeof(Particle), cudaMemcpyHostToDevice);
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    compute_density_and_pressure<<<grid_dim, block_dim>>>(n);

    cudaDeviceSynchronize();

    cudaMemcpy(dst_density, densities, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_pressure, pressures, n * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void compute_pressure_grad_newton(int n) {

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    float2 grad = make_float2(0.0f, 0.0f);
    Particle cur = *(Particle*)&params.particles[index * sizeof(Particle)];
    
    for(int i = 0; i < n; i++) {
        Particle p = *(Particle*)&params.particles[i * sizeof(Particle)];
    
        if(p.id == cur.id)
            continue;
        assert(params.densities[p.id] > 0);

        float2 disp = make_float2(
            cur.pos.x - p.pos.x,
            cur.pos.y - p.pos.y
        );
        
        float pressure = (params.pressures[p.id] + params.pressures[cur.id]) * 0.5f;

        float2 kernel_grad = smoothing_kernal_grad(disp);
        grad = make_float2(
            grad.x + kernel_grad.x * pressure / params.densities[p.id],
            grad.y + kernel_grad.y * pressure / params.densities[p.id]
        );
    }

    params.pressure_grads[index] = grad;

}

void compute_pressure_grads_newton_gpu(int n, Vec2 *dst_grad) {
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    compute_pressure_grad_newton<<<grid_dim, block_dim>>>(n);

    cudaDeviceSynchronize();

    cudaMemcpy(dst_grad, pressure_grads, n * sizeof(float2), cudaMemcpyDeviceToHost);
}

__device__ inline float2 compute_acc(int index) {
    // return pressure_grads[index] * (-1.0 / densities[index]) + Vec2(0.0f, -9.8f);
    float2 grad = params.pressure_grads[index];
    return make_float2(
        grad.x * (-1.0 / params.densities[index]),
        grad.y * (-1.0 / params.densities[index])
    );
}

__global__ void compute_x_dot(int n) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    Particle cur = *(Particle*)&params.particles[index * sizeof(Particle)];
    StateDerivateCUDA *s = (StateDerivateCUDA*)&params.x_dots[index * sizeof(StateDerivateCUDA)];

    s->vel = make_float2(cur.vel.x, cur.vel.y);
    s->acc = compute_acc(index);
}

void compute_x_dot_gpu(int n, StateDerivative *dst_x_dot) {
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    compute_x_dot<<<grid_dim, block_dim>>>(n);

    cudaDeviceSynchronize();

    cudaMemcpy(dst_x_dot, x_dots, n * sizeof(StateDerivative), cudaMemcpyDeviceToHost);
}