#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#include <cassert>
#include <cmath>

struct Vec2 {
    float x;
    float y;
    Vec2(float vx = 0.0f, float vy = 0.0f) : x(vx), y(vy) {};
    float dist2(Vec2 v) {
        return (x - v.x) * (x - v.x) + (y - v.y) * (y - v.y);
    }
    float norm2() {
        return x * x + y * y;
    }
    Vec2 normalize() {
        float norm = sqrt(norm2());
        if(norm == 0.0f) {
            return Vec2(0.0f, 0.0f);
        } else return *this / norm;
    }
    Vec2 operator+(Vec2 v) {
        return Vec2(x + v.x, y + v.y);
    }
    Vec2 operator-(Vec2 v) {
        return Vec2(x - v.x, y - v.y);
    }
    Vec2 operator*(float a) {
        return Vec2(x * a, y * a);
    }
    Vec2 operator/(float a) {
        assert(a != 0.0f);
        return Vec2(x / a, y / a);
    }
    Vec2 rotate(float theta) {
        double rot2d[2][2] = {
            { cos(theta), -sin(theta) }, 
            { sin(theta), cos(theta) }
        };
        return Vec2(
            rot2d[0][0] * x + rot2d[0][1] * y,
            rot2d[1][0] * x + rot2d[1][1] * y
        );
    }
};

struct Particle {
    int id;
    Vec2 pos;
    Vec2 vel;
    int block;
};

struct StateDerivative {
    Vec2 vel;
    Vec2 acc;
};

#endif