#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
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
#include "gpu.h"
#include "bitonic_sort.h"

#define PARTICLES 10
#define PARTICLE_RADIUS 0.1f
#define PARTICLE_TILE_NUMBER 512
#define SAMPLE_TILE_NUMBER 10
#define OCCUPANCY 0.6f
#define BOX_WIDTH 20.0f
#define BOX_HEIGHT 20.0f
#define EPS 1e-3f
#define SMOOTH_RADIUS 1.0f
#define SMOOTH_RADIUS2 (SMOOTH_RADIUS * SMOOTH_RADIUS)
#define SMOOTH_RADIUS4 (SMOOTH_RADIUS2 * SMOOTH_RADIUS2)
#define TWO_THIRDS 2.0f / 3.0f
#define DAMPING_FACTOR 1.0f

#define PRESSURE_RESPONSE 200.0f

#define TEXTURE_SUBDIVS 128

static const int WINDOW_WIDTH = 800;
static const int WINDOW_HEIGHT = 600;

static const float BLOCK_LEN = SMOOTH_RADIUS;
static const int BLOCKS_X = static_cast<int>(std::ceil(BOX_WIDTH / BLOCK_LEN));
static const int BLOCKS_Y = static_cast<int>(std::ceil(BOX_HEIGHT / BLOCK_LEN));

static float momentum = 0.9;
static float running_duration;

std::array<std::array<std::vector<Particle>, BLOCKS_Y>, BLOCKS_X> blocks;

float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
float normalizer = 1 / kernel_volume;

static float average_density = PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER / (BOX_WIDTH * BOX_HEIGHT);
static float desired_density = average_density;

static const float dt = 0.01;

static std::mt19937 gen(114514);
static std::uniform_real_distribution<float> distribution(0, 1);

static float viewport_width;
static float viewport_height;

static std::vector<Particle> particles(PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER);

static int frame = 0;

static void errorCallback(int error, const char *description) {
    std::cerr << "Error: " << description << std::endl;
}

static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static void renderCircle(float x, float y, float radius) {
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

static void drawBox(float x1, float y1, float x2, float y2) {
    glBegin(GL_LINE_LOOP);
    glVertex2f(x1, y1);
    glVertex2f(x2, y1);
    glVertex2f(x2, y2);
    glVertex2f(x1, y2);
    glEnd();
}

static GLFWwindow *create_window() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(1);
    }

    glfwSetErrorCallback(errorCallback);

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

    glfwSetKeyCallback(window, keyCallback);

    return window;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {

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
static void tile_particles(std::vector<Particle> &particles) {
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

static void draw_arrow(Vec2 start, Vec2 disp) {
    // disp = disp.normalize();
    Vec2 end = start + disp;
    static const float theta = 0.3f;
    static const float scale = 0.2f;
    
    Vec2 flap1 = end - disp.rotate(theta) * scale;
    Vec2 flap2 = end - disp.rotate(-theta) * scale;

    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(start.x, start.y, 0.0f);
    glVertex3f(end.x, end.y, 0.0f);
    glEnd();

    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(end.x, end.y, 0.0f);
    glVertex3f(flap1.x, flap1.y, 0.0f);
    glEnd();

    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(end.x, end.y, 0.0f);
    glVertex3f(flap2.x, flap2.y, 0.0f);
    glEnd();
}


static void report_time(Timer &t, const char *str) {
    printf("%s took %f seconds\n", str, t.time());
    t.reset();
}

static void increment_time(Timer &timer, double &acc) {
    acc += timer.time();
    timer.reset();
}

int main() {

    gpu_init(particles.size(), dt, desired_density, BOX_WIDTH, BOX_HEIGHT);

    show_device();

    printf("BLOCK_X: %d, BLOCK_Y: %d\n", BLOCKS_X, BLOCKS_Y);
    
    GLFWwindow *window = create_window();
    framebuffer_size_callback(window, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    tile_particles(particles);

    load_particles_to_gpu(particles.data(), particles.size());

    static double times[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    Timer duration;

    for(int i = 0; i < 1000; i++) {
        
        if(glfwWindowShouldClose(window))
            break;

        duration.reset();

        frame++;

        set_default();

        Timer timer;
        compute_densities_and_pressures_gpu(particles.size());
        increment_time(timer, times[0]);

        compute_pressure_grads_newton_gpu(particles.size());
        increment_time(timer, times[1]);

        compute_x_dot_gpu(particles.size());
        increment_time(timer, times[2]);

        step_ahead_gpu(particles.size());  
        increment_time(timer, times[3]);

        set_altered();

        compute_densities_and_pressures_gpu(particles.size());
        increment_time(timer, times[4]);

        compute_pressure_grads_newton_gpu(particles.size());
        increment_time(timer, times[5]);

        compute_x_dot_gpu(particles.size());
        increment_time(timer, times[6]);

        update_particles_gpu(particles.size(), particles.data());
        increment_time(timer, times[7]);

        // std::string str;
        // std::getline(std::cin, str);

        glClear(GL_COLOR_BUFFER_BIT);   
        for(auto &p: particles) {
            if(p.id == 255)
                glColor3f(1.0f, 0.0f, 0.0f);
            else 
                glColor3f(1.0f, 1.0f, 1.0f);

            renderCircle(p.pos.x, p.pos.y, PARTICLE_RADIUS / 2);
        }
        
        glColor3f(1.0f, 1.0f, 1.0f);
        drawBox(-BOX_WIDTH / 2 + EPS, -BOX_HEIGHT / 2 + EPS, BOX_WIDTH / 2 - EPS, BOX_HEIGHT / 2 - EPS);

        glfwSwapBuffers(window);
        glfwPollEvents();

        running_duration = momentum * running_duration + (1 - momentum) * duration.time();

        if(frame % 100 == 0) {
            printf("\nfps: %.6g\n", 1 / running_duration);
            printf("density & pressure: %.6g\npressure grad: %.6g\nx dot: %.6g\nstep ahead: %.6g\ndensity & pressure: %.6g\npressure grad: %.6g\nx dot: %.6g\nupdate: %.6g\n\n", times[0] / frame, times[1] / frame, times[2] / frame, times[3] / frame, times[4] / frame, times[5] / frame, times[6] / frame, times[7] / frame);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}