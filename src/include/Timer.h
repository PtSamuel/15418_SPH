#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double time() {
        return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(std::chrono::high_resolution_clock::now() - start).count();
    }
    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }
};

#endif