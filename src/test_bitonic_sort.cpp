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
        // particles[i].block = i;
    }

    particles[15].block = 15;
    for(int i = 0; i < test_length; i++) {
        printf("%d: %d\n", particles[i].id, particles[i].block);
    }

    bitonic_sort(particles.data(), test_length);

    // for(int i = 0; i < test_length; i++) {
    //     printf("%d: %d\n", particles[i].id, particles[i].block);
    // }

    return 0;
}