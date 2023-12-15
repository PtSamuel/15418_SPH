#include "Particle.h"
#include "bitonic_sort.h"
#include <vector>
#include <iostream>

int main() {
    const int test_length = 16;
    std::vector<Particle> particles(test_length);
    for(int i = 0; i < test_length; i++) {
        particles[i].id = i;
        particles[i].block = test_length - 1 - i;
    }

    particles[2].block = 12;
    particles[9].block = 12;
    particles[15].block = 12;
    particles[11].block = 12;

    printf("before:\n");
    for(int i = 0; i < test_length; i++) {
        printf("%d: %d\n", particles[i].id, particles[i].block);
    }

    bitonic_sort_cpu(particles.data(), test_length);

    printf("after:\n");
    for(int i = 0; i < test_length; i++) {
        printf("%d: %d\n", particles[i].id, particles[i].block);
    }

    return 0;
}