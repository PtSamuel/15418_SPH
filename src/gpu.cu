#include <string>
#include <math.h>
#include <cuda_runtime.h>

#include "gpu.h"
#include "Particle.h"
#include "Timer.h"
#include "bitonic_sort.h"

#define BLOCK_DIM 16
#define THREADS_PER_BLOCK (BLOCK_DIM * BLOCK_DIM)

#define TWO_THIRDS 2.0f / 3.0f

#define SMOOTH_RADIUS 1.0f
#define SMOOTH_RADIUS2 (SMOOTH_RADIUS * SMOOTH_RADIUS)
#define SMOOTH_RADIUS4 (SMOOTH_RADIUS2 * SMOOTH_RADIUS2)

#define PRESSURE_RESPONSE 200.0f

static const float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
static const float normalizer = 1 / kernel_volume;

static float block_len;
static int blocks_x;
static int blocks_y;

struct StateDerivateCUDA {
    float2 vel;
    float2 acc;
};

enum SwapStatus {
    SWAP_DEFAULT,
    SWAP_ALTERED
};
static SwapStatus status;

void set_default() {
    status = SWAP_DEFAULT;
}

void set_altered() {
    status = SWAP_ALTERED;
}

static uchar1 *particles;
static uchar1 *particles_swap;
static float *densities;
static float *pressures;
static float2 *pressure_grads;
static StateDerivateCUDA *x_dots;

static Particle *blocks;
static int *block_size_lookup;

static int *dividers;

struct CUDAParams {
    float dt;
    uchar1 *particles;
    uchar1 *particles_swap;
    float *densities;
    float *pressures;
    float2 *pressure_grads;
    StateDerivateCUDA *x_dots;

    Particle *blocks;
    int *block_size_lookup;

    int *dividers;

    float desired_density;
    float block_len;
    int blocks_x;
    int blocks_y;

    float box_width;
    float box_height;

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

void gpu_init(int n, float step, float desired_density, float w, float h) {

    block_len = SMOOTH_RADIUS;
    blocks_x = static_cast<int>(std::ceil(w / block_len));
    blocks_y = static_cast<int>(std::ceil(h / block_len));

    cudaMalloc(&particles, n * sizeof(Particle));
    cudaMalloc(&particles_swap, n * sizeof(Particle));
    cudaMalloc(&densities, n * sizeof(float));
    cudaMalloc(&pressures, n * sizeof(float));
    cudaMalloc(&pressure_grads, n * sizeof(float2));
    cudaMalloc(&x_dots, n * sizeof(StateDerivateCUDA));
    cudaMalloc(&dividers, blocks_x * blocks_y * sizeof(int));

    CUDAParams p;
    p.dt = step;
    p.particles = particles;
    p.particles_swap = particles_swap;
    p.densities = densities;
    p.pressures = pressures;
    p.pressure_grads = pressure_grads;
    p.desired_density = desired_density;
    p.x_dots = x_dots;

    p.box_width = w;
    p.box_height = h;

    status = SWAP_DEFAULT;

    p.block_len = block_len;
    p.blocks_x = blocks_x;
    p.blocks_y = blocks_y;

    int num_partitions = blocks_x * blocks_y;
    cudaMalloc(&blocks, n * sizeof(Particle) * num_partitions);
    cudaMalloc(&block_size_lookup, num_partitions * sizeof(int));

    p.blocks = blocks;
    p.block_size_lookup = block_size_lookup;
    p.dividers = dividers;

    cudaMemcpyToSymbol(params, &p, sizeof(CUDAParams));
}

void load_particles_to_gpu(Particle *p, int n) {
    cudaMemcpy(particles, p, sizeof(Particle) * n, cudaMemcpyHostToDevice);
}

__device__ inline uint2 get_block(Vec2 pos) {
    uint x = (pos.x + params.box_width / 2) / params.block_len;
    uint y = (pos.y + params.box_height / 2) / params.block_len;
    x = min(x, params.blocks_x - 1);
    y = min(y, params.blocks_y - 1);
    return make_uint2(x, y);
}

__global__ void compute_particle_block(int n, Particle* particles) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;

    if(index >= n) return;

    Particle &p = particles[index];

    uint2 coords = get_block(p.pos);
    p.block = coords.y * params.blocks_x + coords.x;
}

__global__ void print_particles(int n, Particle* particles) {
    for(int i = 0; i < n; i++) {
        printf("particle %d: (%f, %f), (%f, %f) block = %d\n", 
            particles[i].id,
            particles[i].pos.x, particles[i].pos.y,
            particles[i].vel.x, particles[i].vel.x, 
            particles[i].block);
    }
}

__global__ void init_divider(int num_find_divider_tasks) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= num_find_divider_tasks) return;
    params.dividers[index] = -1;
}


__global__ void find_divider(int n, Particle *particles) {

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;

    if(index >= n) return;

    int blockcur = particles[index].block;
    if(index == 0)
        params.dividers[blockcur] = index;
    else {
        int blockprev = particles[index - 1].block;
        if(blockprev < blockcur)
            params.dividers[blockcur] = index;
    }
    
}

static void init_dividers() {
    int num_find_divider_tasks = blocks_x * blocks_y;
    
    int num_blocks = (num_find_divider_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    init_divider<<<grid_dim, block_dim>>>(num_find_divider_tasks);
    cudaDeviceSynchronize();
}

static void find_dividers(int n, Particle* particles) {

    init_dividers();
    int num_find_divider_tasks = n;
    
    int num_blocks = (num_find_divider_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    find_divider<<<grid_dim, block_dim>>>(n, (Particle*)particles);
    cudaDeviceSynchronize();
}

static void report_time(Timer &t, const char *str) {
    printf("%s took %f seconds\n", str, t.time());
    t.reset();
}

void partition_particles(int n) {

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    // Timer timer;

    if(status == SWAP_DEFAULT) {

        // mess<<<1, 1>>>(n, (Particle*)particles);
        // cudaDeviceSynchronize();

        // printf("before:\n");
        // print_particles<<<1, 1>>>(n, (Particle*)particles);
        // cudaDeviceSynchronize();

        compute_particle_block<<<grid_dim, block_dim>>>(n, (Particle*)particles);
        cudaDeviceSynchronize();

        // report_time(timer, "find blocks");

        bitonic_sort((Particle*)particles, n);

        // report_time(timer, "bitonic sort");

        // printf("after:\n");
        // print_particles<<<1, 1>>>(n, (Particle*)particles);
        // cudaDeviceSynchronize();
        
        find_dividers(n, (Particle*)particles);

        // report_time(timer, "find dividers");


    } else {


        // mess<<<1, 1>>>(n, (Particle*)particles);
        // cudaDeviceSynchronize();

        // printf("before:\n");
        // print_particles<<<1, 1>>>(n, (Particle*)particles_swap);
        // cudaDeviceSynchronize();

        compute_particle_block<<<grid_dim, block_dim>>>(n, (Particle*)particles_swap);
        cudaDeviceSynchronize();
        bitonic_sort((Particle*)particles_swap, n);
        find_dividers(n, (Particle*)particles_swap);

    }

}

__global__ void compute_density_and_pressure(int n, Particle *particles) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    Particle cur = particles[index];
    uint2 coords = get_block(cur.pos);

    float2 pos = make_float2(
        cur.pos.x,
        cur.pos.y
    );
    
    float density = 0;

    for(int y = (int)coords.y - 1; y <= (int)coords.y + 1; y++)
        for(int x = (int)coords.x - 1; x <= (int)coords.x + 1; x++) {
            
            if(x < 0 || x >= params.blocks_x || y < 0 || y >= params.blocks_y)
                continue;

            int block_index = y * params.blocks_x + x;
            int divider = params.dividers[block_index];
            if(divider == -1)
                continue;

            for(int i = divider; i < n; i++) {
                Particle p = particles[i];
                
                if(p.block != block_index)
                    break;
                    
                float2 disp = make_float2(
                    pos.x - p.pos.x,
                    pos.y - p.pos.y
                );

                density += smoothing_kernal(disp);
            }
        }

    params.densities[cur.id] = density;
    
    float pressure = PRESSURE_RESPONSE * (density - params.desired_density);
    params.pressures[cur.id] = pressure;
}

static void increment_time(Timer &timer, double &acc) {
    acc += timer.time();
    timer.reset();
}

void compute_densities_and_pressures_gpu(int n) {

    static int frame = 0;
    static double times[2] = { 0.0, 0.0 };
    Timer timer;

    partition_particles(n);

    increment_time(timer, times[0]);

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    if(status == SWAP_DEFAULT)
        compute_density_and_pressure<<<grid_dim, block_dim>>>(n, (Particle*)particles);
    else compute_density_and_pressure<<<grid_dim, block_dim>>>(n, (Particle*)particles_swap);

    cudaDeviceSynchronize();

    increment_time(timer, times[1]);
    
    frame++;
    if(frame % 200 == 0) {
        printf("gpu: partition: %.6g\ndensity & pressure: %.6g\n", times[0] / frame, times[1] / frame);
    }

}

__global__ void compute_pressure_grad_newton(int n, Particle *particles) {

    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    float2 grad = make_float2(0.0f, 0.0f);

    Particle cur = particles[index];

    uint2 coords = get_block(cur.pos);
    
    int xstart = max(0, (int)coords.x - 1);
    int xend = min(params.blocks_x - 1, (int)coords.x + 1);

    int ystart = max(0, (int)coords.y - 1);
    int yend = min(params.blocks_y - 1, (int)coords.y + 1);

    for(int y = ystart; y <= yend; y++) {
        int blockstart = y * params.blocks_x + xstart;
        int blockend = y * params.blocks_x + xend;
        
        int start = blockstart;
        if(params.dividers[start] == -1)
            start++;
        if(params.dividers[start] == -1)
            start++;
        if(params.dividers[start] == -1)
            start++;
        
        if(start > blockend)
            continue;

        for(int i = params.dividers[start]; i < n; i++) {
            if(i == index) continue;

            Particle p = particles[i];
            if(p.block > blockend) {
                break;
            }

            float2 disp = make_float2(
                cur.pos.x - p.pos.x,
                cur.pos.y - p.pos.y
            );
            
            float pressure = (params.pressures[cur.id] + params.pressures[p.id]) * 0.5f;

            float2 kernel_grad = smoothing_kernal_grad(disp);
            grad = make_float2(
                grad.x + kernel_grad.x * pressure / params.densities[p.id],
                grad.y + kernel_grad.y * pressure / params.densities[p.id]
            );
        }
    }

    params.pressure_grads[cur.id] = grad;

}

void compute_pressure_grads_newton_gpu(int n) {

    Timer timer;
    static int frame = 0;
    static double time = 0.0;

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    if(status == SWAP_DEFAULT)
        compute_pressure_grad_newton<<<grid_dim, block_dim>>>(n, (Particle*)particles);
    else compute_pressure_grad_newton<<<grid_dim, block_dim>>>(n, (Particle*)particles_swap);

    cudaDeviceSynchronize();

    increment_time(timer, time);
    frame++;
    if(frame % 200 == 0) {
        printf("gpu: pressure grad: %.6g\n", time / frame);
    }   
}

__device__ inline float2 compute_acc(int id) {
    float2 grad = params.pressure_grads[id];
    return make_float2(
        grad.x * (-1.0 / params.densities[id]),
        grad.y * (-1.0 / params.densities[id])
    );
}

__global__ void compute_x_dot(int n, SwapStatus status) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    if(status == SWAP_DEFAULT) {
        Particle *particles = (Particle*)params.particles;
        Particle cur = particles[index];
        params.x_dots[cur.id].vel = make_float2(cur.vel.x, cur.vel.y);
        params.x_dots[cur.id].acc = compute_acc(cur.id);
    } else {
        Particle *particles = (Particle*)params.particles_swap;
        Particle cur = particles[index];
        float2 vel = make_float2(cur.vel.x, cur.vel.y);
        float2 acc = compute_acc(cur.id);
        
        StateDerivateCUDA x_dot = params.x_dots[cur.id];
        StateDerivateCUDA updated_x_dot;

        // RK2
        updated_x_dot.vel.x = x_dot.vel.x * 0.25 + vel.x * 0.75;
        updated_x_dot.vel.y = x_dot.vel.y * 0.25 + vel.y * 0.75;

        updated_x_dot.acc.x = x_dot.acc.x * 0.25 + acc.x * 0.75;
        updated_x_dot.acc.y = x_dot.acc.y * 0.25 + acc.y * 0.75;

        // SIMPLER LOOKAHEAD
        // updated_x_dot.vel.x = x_dot.vel.x;
        // updated_x_dot.vel.y = x_dot.vel.y;

        // updated_x_dot.acc.x = acc.x;
        // updated_x_dot.acc.y = acc.y;

        // LEAPFROG
        // updated_x_dot.vel.x = x_dot.vel.x + acc.x * params.dt;
        // updated_x_dot.vel.y = x_dot.vel.y + acc.x * params.dt;

        // updated_x_dot.acc.x = acc.x;
        // updated_x_dot.acc.y = acc.y;

        params.x_dots[cur.id] = updated_x_dot;
    }
}

void compute_x_dot_gpu(int n) {

    Timer timer;
    static int frame = 0;
    static double time = 0.0;

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    compute_x_dot<<<grid_dim, block_dim>>>(n, status);
    cudaDeviceSynchronize();

    increment_time(timer, time);
    frame++;
    if(frame % 200 == 0) {
        printf("gpu: compute derivative: %.6g\n", time / frame);
    }   
}

__global__ void step_ahead(int n, Particle *particles, Particle *update) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    Particle cur = particles[index];

    cur.pos.x += params.x_dots[cur.id].vel.x * params.dt * TWO_THIRDS;
    cur.pos.y += params.x_dots[cur.id].vel.y * params.dt * TWO_THIRDS;
    cur.vel.x += params.x_dots[cur.id].acc.x * params.dt * TWO_THIRDS;
    cur.vel.y += params.x_dots[cur.id].acc.y * params.dt * TWO_THIRDS;

    // LEAPFROG
    // cur.pos.x += params.x_dots[index].vel.x * params.dt * 0.5;
    // cur.pos.y += params.x_dots[index].vel.y * params.dt * 0.5;
    // cur.vel.x += params.x_dots[index].acc.x * params.dt * 0.5;
    // cur.vel.y += params.x_dots[index].acc.y * params.dt * 0.5;
    
    update[index] = cur;
}

void step_ahead_gpu(int n) {

    Timer timer;
    static int frame = 0;
    static double time = 0.0;

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    step_ahead<<<grid_dim, block_dim>>>(n, (Particle*)particles, (Particle*)particles_swap);

    cudaDeviceSynchronize();

    increment_time(timer, time);
    frame++;
    if(frame % 100 == 0) {
        printf("gpu: step ahead: %.6g\n", time / frame);
    }   
}

__device__ inline void clamp_particle(Particle &p) {

    float box_width = params.box_width;
    float box_height = params.box_height;

    if(p.pos.x > box_width / 2) {
        p.pos.x = box_width - p.pos.x;
        p.pos.x = fmax(p.pos.x, -box_width / 2);
        p.vel.x = -fabs(p.vel.x);
    } else if(p.pos.x < -box_width / 2) {
        p.pos.x = -box_width - p.pos.x;
        p.pos.x = fmin(p.pos.x, box_width / 2);
        p.vel.x = fabs(p.vel.x);
    }

    if(p.pos.y > box_height / 2) {
        p.pos.y = box_height - p.pos.y;
        p.pos.y = fmax(p.pos.y, -box_height / 2);
        p.vel.y = -fabs(p.vel.y);
    } else if(p.pos.y < -box_height / 2) {
        p.pos.y = -box_height - p.pos.y;
        p.pos.y = fmin(p.pos.y, box_height / 2);
        p.vel.y = fabs(p.vel.y);
    }
}

__global__ void update_particle(int n, Particle *particles) {
    
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y * BLOCK_DIM + threadIdx.x;
    if(index >= n) return;

    Particle p = particles[index];
        
    float dt = params.dt;
    
    // RK2
    p.pos.x += params.x_dots[p.id].vel.x * dt + params.x_dots[p.id].acc.x * dt * dt * 0.5;
    p.pos.y += params.x_dots[p.id].vel.y * dt + params.x_dots[p.id].acc.y * dt * dt * 0.5;

    p.vel.x += params.x_dots[p.id].acc.x * dt;
    p.vel.y += params.x_dots[p.id].acc.y * dt;

    // SIMPLER LOOKAHEAD
    // p.vel.x += params.x_dots[index].acc.x * dt;
    // p.vel.y += params.x_dots[index].acc.y * dt;

    // p.pos.x += p.vel.x * dt;
    // p.pos.y += p.vel.y * dt;

    // LEAPFROG
    // Vec2 vel_prev = p.vel;
    // p.vel.x += params.x_dots[index].acc.x * dt;
    // p.vel.y += params.x_dots[index].acc.y * dt;

    // p.pos.x += (vel_prev.x + p.vel.x) * 0.5 * dt;
    // p.pos.y += (vel_prev.y + p.vel.y) * 0.5 * dt;

    clamp_particle(p);
    
    particles[index] = p;
}

void update_particles_gpu(int n, Particle *dst_particles_swap) {

    Timer timer;
    static int frame = 0;
    static double times[] = { 0.0, 0.0 } ;

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    update_particle<<<grid_dim, block_dim>>>(n, (Particle*)particles);

    cudaDeviceSynchronize();

    increment_time(timer, times[0]);

    cudaMemcpy(dst_particles_swap, particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    increment_time(timer, times[1]);

    frame++;
    if(frame % 100 == 0) {
        printf("gpu: update particles: %.6g\ncopy to cpu: %.6g\n", times[0] / frame, times[1] / frame);
    }   
}