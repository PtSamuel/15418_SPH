#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>
#include <array>
#include <cassert>
#include <set>

#include "Particle.h"
#include "Timer.h"

#define PARTICLE_RADIUS 0.1f
#define PARTICLE_TILE_NUMBER 64
#define OCCUPANCY 0.5f
#define BOX_WIDTH 40.0f
#define BOX_HEIGHT 40.0f
#define EPS 1e-3f
#define SMOOTH_RADIUS 1.0f
#define SMOOTH_RADIUS2 (SMOOTH_RADIUS * SMOOTH_RADIUS)
#define SMOOTH_RADIUS4 (SMOOTH_RADIUS2 * SMOOTH_RADIUS2)
#define TWO_THIRDS 2.0f / 3.0f
#define DAMPING_FACTOR 1.0f

#define PRESSURE_RESPONSE 200.0f

static const int WINDOW_WIDTH = 1848;
static const int WINDOW_HEIGHT = 1016;

const float BLOCK_LEN = SMOOTH_RADIUS;
const int BLOCKS_X = static_cast<int>(std::ceil(BOX_WIDTH / BLOCK_LEN));
const int BLOCKS_Y = static_cast<int>(std::ceil(BOX_HEIGHT / BLOCK_LEN));

float momentum = 0.9;
float running_duration;

std::array<std::array<std::vector<Particle>, BLOCKS_Y>, BLOCKS_X> blocks;

float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
float normalizer = 1 / kernel_volume;

float average_density = PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER / (BOX_WIDTH * BOX_HEIGHT);
float desired_density = average_density;

const float dt = 0.01;

float viewport_width;
float viewport_height;

std::vector<Particle> particles(PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER);

std::vector<float> densities(particles.size());
std::vector<float> pressures(particles.size());
std::vector<Vec2> pressure_grads(particles.size());
std::vector<StateDerivative> x_dots(particles.size());

std::vector<Particle> particles_swap(particles.size());

std::vector<StateDerivative> x_dots_swap(particles.size());

float max_density;

int frame = 0;

static void error_callback(int error, const char *description) {
    std::cerr << "Error: " << description << std::endl;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static void render_circle(float x, float y, float radius) {
    const int sides = 100;

    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y); // Center of the circle

    for (int i = 0; i <= sides; ++i) {
        float angle = 2.0f * M_PI * i / sides;
        float x_cur = x + radius * std::cos(angle);
        float y_cur = y + radius * std::sin(angle);
        glVertex2f(x_cur, y_cur);
    }

    glEnd();
}

static void draw_box(float x1, float y1, float x2, float y2) {
    glBegin(GL_LINE_LOOP);
    glVertex2f(x1, y1);
    glVertex2f(x2, y1);
    glVertex2f(x2, y2);
    glVertex2f(x1, y2);
    glEnd();
}

GLFWwindow *create_window() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(1);
    }

    glfwSetErrorCallback(error_callback);

    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "OpenGL Circle", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(1);
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        exit(1);
    }

    glfwSetKeyCallback(window, key_callback);

    return window;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {

    float new_width = width, new_height = height;

    float ratio = (float)width / height;
    float box_ratio = BOX_WIDTH / BOX_HEIGHT;

    if(box_ratio > ratio)
        new_height = (float)width / box_ratio;
    else
        new_width = (float)height * box_ratio;

    int padding = 50;
    new_width -= padding * 2 * box_ratio;
    new_height -= padding * 2;

    int bx = (width - new_width) / 2;
    int by = (height - new_height) / 2;

    glViewport(bx, by, new_width, new_height);

    float viewport_padding_y = BOX_HEIGHT * 0.1;
    float viewport_padding_x = BOX_WIDTH / BOX_HEIGHT * viewport_padding_y;

    viewport_width = BOX_WIDTH + 2 * viewport_padding_x;
    viewport_height = BOX_HEIGHT + 2 * viewport_padding_y;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(
        -viewport_width / 2, 
        viewport_width / 2, 
        -viewport_height / 2, 
        viewport_height / 2, 
        -1.0, 1.0
    );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void tile_particles(std::vector<Particle> &particles) {
    assert(particles.size() == PARTICLE_TILE_NUMBER* PARTICLE_TILE_NUMBER);
    for(int j = 0; j < PARTICLE_TILE_NUMBER; j++)
        for(int i = 0; i < PARTICLE_TILE_NUMBER; i++)
        {
            auto &p = particles[PARTICLE_TILE_NUMBER * j + i];
            p.id = PARTICLE_TILE_NUMBER * j + i;
            p.pos.x = (float)(i - PARTICLE_TILE_NUMBER * 0.5) / PARTICLE_TILE_NUMBER * OCCUPANCY * BOX_WIDTH;
            p.pos.y = (float)(j - PARTICLE_TILE_NUMBER * 0.5) / PARTICLE_TILE_NUMBER * OCCUPANCY * BOX_HEIGHT;
        }
}

float smoothing_kernal(Vec2 disp) {
    float dist = sqrt(disp.norm2());
    float offset = std::max(0.0f, SMOOTH_RADIUS - dist);
    return offset * offset * normalizer;
}

Vec2 smoothing_kernal_grad(Vec2 disp) {
    float dist2 = disp.norm2();
    if(dist2 == 0.0f || dist2 > SMOOTH_RADIUS2)
        return Vec2(0, 0);
    
    float dist = sqrt(dist2);

    float x = -2 * (SMOOTH_RADIUS - dist) * disp.x / dist * normalizer;
    float y = -2 * (SMOOTH_RADIUS - dist) * disp.y / dist * normalizer;
    return Vec2(x, y);
}

std::pair<int, int> get_block(Vec2 pos) {
    int x = (pos.x + BOX_WIDTH / 2) / BLOCK_LEN;
    int y = (pos.y + BOX_HEIGHT / 2) / BLOCK_LEN;
    x = std::min(x, BLOCKS_X - 1);
    y = std::min(y, BLOCKS_Y - 1);
    return { x, y };
}

void distribute() {
    for(int i = 0; i < BLOCKS_X; i++) 
        for(int j = 0; j < BLOCKS_Y; j++)
            blocks[i][j].clear();
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        auto coords = get_block(p.pos);
        blocks[coords.first][coords.second].push_back(p);
    }
}

float compute_density(Vec2 pos) {
    auto coords = get_block(pos);
    int x = coords.first, y = coords.second;

    float density = 0;

    for(int i = x - 1; i <= x + 1; i++)
        for(int j = y - 1; j <= y + 1; j++) {
            if(i < 0 || i >= BLOCKS_X || j < 0 || j >= BLOCKS_Y)
                continue;
            for(auto &p: blocks[i][j]) {
                Vec2 disp = pos - p.pos;
                density += smoothing_kernal(disp);
            }
        }

    return density;
}

void compute_densities() {
    float max = 0;
    for(int i = 0; i < particles.size(); i++) {
        float density = compute_density(particles[i].pos);
        densities[i] = density;
        if(density > max)
            max = density;
    }
    max_density = max;
}

float compute_pressure(float density) {
    return PRESSURE_RESPONSE * (density - desired_density);
}

void compute_pressures() {
    for(int i = 0; i < particles.size(); i++)
        pressures[i] = compute_pressure(densities[i]);
}

Vec2 compute_pressure_grad_newton(int index) {
    Vec2 grad = Vec2(0.0f, 0.0f);
    Vec2 pos = particles[index].pos;

    auto coords = get_block(pos);
    int x = coords.first, y = coords.second;
    for(int i = x - 1; i <= x + 1; i++)
        for(int j = y - 1; j <= y + 1; j++) {
            if(i < 0 || i >= BLOCKS_X || j < 0 || j >= BLOCKS_Y)
                continue;
            
            for(auto &p: blocks[i][j]) {
                if(p.id == index)
                    continue;
                assert(densities[p.id] > 0);

                float pressure = (pressures[p.id] + pressures[index]) * 0.5f;

                Vec2 kernel_grad = smoothing_kernal_grad(pos - p.pos);
                grad = grad + kernel_grad * (pressure / densities[p.id]);
            }
        }

    return grad;

}

void compute_pressure_grads_newton() {
    for(int i = 0; i < particles.size(); i++)
        pressure_grads[i] = compute_pressure_grad_newton(i);
}

inline void clamp_particle(Particle &p) {
    if(p.pos.x > BOX_WIDTH / 2) {
        p.pos.x = BOX_WIDTH - p.pos.x;
        p.pos.x = std::max(p.pos.x, -BOX_WIDTH / 2);
        p.vel.x = -std::abs(p.vel.x) * DAMPING_FACTOR;
    } else if(p.pos.x < -BOX_WIDTH / 2) {
        p.pos.x = -BOX_WIDTH - p.pos.x;
        p.pos.x = std::min(p.pos.x, BOX_WIDTH / 2);
        p.vel.x = std::abs(p.vel.x) * DAMPING_FACTOR;
    }

    if(p.pos.y > BOX_HEIGHT / 2) {
        p.pos.y = BOX_HEIGHT - p.pos.y;
        p.pos.y = std::max(p.pos.y, -BOX_HEIGHT / 2);
        p.vel.y = -std::abs(p.vel.y) * DAMPING_FACTOR;
    } else if(p.pos.y < -BOX_HEIGHT / 2) {
        p.pos.y = -BOX_HEIGHT - p.pos.y;
        p.pos.y = std::min(p.pos.y, BOX_HEIGHT / 2);
        p.vel.y = std::abs(p.vel.y) * DAMPING_FACTOR;
    }
}

void update_velocities() {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        
        p.pos = p.pos + x_dots[i].vel * dt + x_dots[i].acc * dt * dt * 0.5;
        p.vel = p.vel + x_dots[i].acc * dt;

        clamp_particle(p);
    }
}


inline Vec2 compute_acc(int index) {
    return pressure_grads[index] * (-1.0 / densities[index]);
}

void step_ahead() {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        particles_swap[i].pos = p.pos + x_dots[i].vel * dt * TWO_THIRDS;
        particles_swap[i].vel = p.vel + x_dots[i].acc * dt * TWO_THIRDS;
    }
}

void compute_x_dot() {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        x_dots[i].vel = p.vel;
        x_dots[i].acc = compute_acc(i);
    }
}

void increment_x_dot(float cur_weight) {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        StateDerivative s;
        s.vel = p.vel;
        s.acc = compute_acc(i);
        x_dots[i].vel = s.vel * cur_weight + x_dots[i].vel * (1 - cur_weight);
        x_dots[i].acc = s.acc * cur_weight + x_dots[i].acc * (1 - cur_weight);
    }
}

static void increment_time(Timer &timer, double &acc) {
    acc += timer.time();
    timer.reset();
}

int main() {

    printf("BLOCK_X: %d, BLOCK_Y: %d\n", BLOCKS_X, BLOCKS_Y);
    
    GLFWwindow *window = create_window();
    framebuffer_size_callback(window, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    tile_particles(particles);
    particles_swap = particles;

    Timer duration;
    // distribute, density & pressure, grad, x dot, update 
    static double times[] = { 0, 0, 0, 0, 0 };

    while(true) {

        duration.reset();

        frame++;

        /* NEW WAY SEEMS TO WORK */ 
        Timer timer;

        distribute();
        increment_time(timer, times[0]);

        compute_densities();
        increment_time(timer, times[1]);

        compute_pressures();
        increment_time(timer, times[1]);

        compute_pressure_grads_newton();
        increment_time(timer, times[2]);

        compute_x_dot();
        increment_time(timer, times[3]);

        step_ahead();
        particles.swap(particles_swap);
        increment_time(timer, times[4]);

        distribute();
        increment_time(timer, times[0]);

        compute_densities();
        increment_time(timer, times[1]);

        compute_pressures();
        increment_time(timer, times[1]);

        compute_pressure_grads_newton();
        increment_time(timer, times[2]);

        increment_x_dot(0.75);
        particles.swap(particles_swap);
        increment_time(timer, times[3]);

        update_velocities();
        increment_time(timer, times[4]);


        glClear(GL_COLOR_BUFFER_BIT);   
        glColor3f(1.0f, 1.0f, 1.0f);
        for(auto &p: particles)
            render_circle(p.pos.x, p.pos.y, PARTICLE_RADIUS);
        
        draw_box(-BOX_WIDTH / 2 + EPS, -BOX_HEIGHT / 2 + EPS, BOX_WIDTH / 2 - EPS, BOX_HEIGHT / 2 - EPS);

        glfwSwapBuffers(window);
        glfwPollEvents();

        running_duration = momentum * running_duration + (1 - momentum) * duration.time();

        if(frame % 4 == 0) {
            printf("fps: %f\n", 1 / running_duration);
            printf("distribute: %.6g\ndensity & pressure: %.6g\npressure grad: %.6g\nx dot: %.6g\nupdate: %.6g\n\n", times[0] / frame, times[1] / frame, times[2] / frame, times[3] / frame, times[4] / frame);
        }

    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}