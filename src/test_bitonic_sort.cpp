#include "Particle.h"
#include "bitonic_sort.h"
#include <vector>

int main() {
    const int test_length = 16;
    std::vector<Particle> particles(test_length);
    for(int i = 0; i < test_length; i++) {
        particles[i].id = i;
        particles[i].block = test_length - 1 - i;
    }

    bitonic_sort(particles.data(), test_length);
    return 0;
}