#ifndef __GPU_H__
#define __GPU_H__

#include "Particle.h"

void show_device();
void gpu_init(int n, float volume);
void compute_densities_gpu(Particle *p, int n, float* dst);

#endif