#ifndef __GPU_H__
#define __GPU_H__

#include "Particle.h"

void show_device();
void gpu_init(int n, float step, float volume);
void load_particles_to_gpu(Particle *p, int n);
void compute_densities_and_pressures_gpu(Particle *p, int n, float* dst_density, float *dst_pressure);
void compute_pressure_grads_newton_gpu(int n, Vec2* dst_pressure_grad);
void compute_x_dot_gpu(int n, StateDerivative *dst_x_dot);
void step_ahead_gpu(int n, Particle *dst_particles_swap);

#endif