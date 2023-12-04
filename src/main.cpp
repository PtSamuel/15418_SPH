#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <Particle.h>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>
#include <Timer.h>

#define PARTICLES 10
#define PARTICLE_RADIUS 0.1f
#define PARTICLE_TILE_NUMBER 20
#define SAMPLE_TILE_NUMBER 10
#define OCCUPANCY 0.8
#define BOX_WIDTH 20.0f
#define BOX_HEIGHT 20.0f
#define EPS 1e-3f
#define SMOOTH_RADIUS 1.0f
#define SMOOTH_RADIUS2 SMOOTH_RADIUS * SMOOTH_RADIUS
#define SMOOTH_RADIUS4 SMOOTH_RADIUS2 * SMOOTH_RADIUS2

#define PRESSURE_RESPONSE 100.0f

#define TEXTURE_SUBDIVS 128

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// float SMOOTH_RADIUS8 = SMOOTH_RADIUS4 * SMOOTH_RADIUS4;
// float kernel_volume = M_PI / 4 * SMOOTH_RADIUS8;
// float normalizer = 1 / kernel_volume;

float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
float normalizer = 1 / kernel_volume;

float average_density = PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER / (BOX_WIDTH * BOX_HEIGHT);
float desired_density = average_density;

const float dt = 0.01;

static std::mt19937 gen(114514);
static std::uniform_real_distribution<float> distribution(0, 1);

std::vector<Particle> particles(PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER);
std::vector<Vec2> kernel_grads(particles.size());
std::vector<float> densities(particles.size());
std::vector<Vec2> density_grads(particles.size());
std::vector<float> pressures(particles.size());
std::vector<Vec2> pressure_grads(particles.size());

std::vector<Particle> particles_swap(particles.size());
std::vector<Vec2> kernel_grads_swap(particles.size());
std::vector<float> densities_swap(particles.size());
std::vector<Vec2> density_grads_swap(particles.size());
std::vector<float> pressures_swap(particles.size());
std::vector<Vec2> pressure_grads_swap(particles.size());

float max_density;

void errorCallback(int error, const char *description) {
    std::cerr << "Error: " << description << std::endl;
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void renderCircle(float x, float y, float radius) {
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

void drawBox(float x1, float y1, float x2, float y2) {
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

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-BOX_WIDTH / 2.0, BOX_WIDTH / 2.0, -BOX_HEIGHT / 2.0, BOX_HEIGHT / 2.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

void init_particles(std::vector<Particle> &particles) {
    for(int i = 0; i < particles.size(); i++) {
        auto &p = particles[i];
        p.id = i;
        p.pos.x = (distribution(gen) - 0.5) * BOX_WIDTH;
        p.pos.y = (distribution(gen) - 0.5) * BOX_HEIGHT;
        p.vel = {0, 0};
    }
}

void tile_particles(std::vector<Particle> &particles) {
    assert(particles.size() == PARTICLE_TILE_NUMBER* PARTICLE_TILE_NUMBER);
    for(int j = 0; j < PARTICLE_TILE_NUMBER; j++)
        for(int i = 0; i < PARTICLE_TILE_NUMBER; i++)
        {
            auto &p = particles[PARTICLE_TILE_NUMBER * j + i];
            p.pos.x = (float)(i - PARTICLE_TILE_NUMBER * 0.5) / PARTICLE_TILE_NUMBER * OCCUPANCY * BOX_WIDTH;
            p.pos.y = (float)(j - PARTICLE_TILE_NUMBER * 0.5) / PARTICLE_TILE_NUMBER * OCCUPANCY * BOX_HEIGHT;
        }
}

// float smoothing_kernal(Vec2 disp) {
//     float dist2 = disp.norm2();
//     float influence = std::max(0.0f, SMOOTH_RADIUS2 - dist2);
//     return influence * influence * influence * normalizer;
// }

// FIX THESE

float smoothing_kernal(Vec2 disp) {
    float dist = sqrt(disp.norm2());
    float offset = std::max(0.0f, SMOOTH_RADIUS - dist);
    return offset * offset * normalizer;
}

// Vec2 smoothing_kernal_grad(Vec2 disp) {
//     float dist2 = disp.norm2();
//     if(dist2 > SMOOTH_RADIUS2)
//         return Vec2(0, 0);
    
//     float x = -3 * (SMOOTH_RADIUS2 - dist2) * (SMOOTH_RADIUS2 - dist2) * disp.x * normalizer;
//     float y = -3 * (SMOOTH_RADIUS2 - dist2) * (SMOOTH_RADIUS2 - dist2) * disp.y * normalizer;
//     return Vec2(x, y);
// }

Vec2 smoothing_kernal_grad(Vec2 disp) {
    float dist2 = disp.norm2();
    if(dist2 == 0.0f || dist2 > SMOOTH_RADIUS2)
        return Vec2(0, 0);
    
    float dist = sqrt(dist2);

    float x = -2 * (SMOOTH_RADIUS - dist) * disp.x / dist * normalizer;
    float y = -2 * (SMOOTH_RADIUS - dist) * disp.y / dist * normalizer;
    return Vec2(x, y);
}

float compute_density(Vec2 pos) {
    float density = 0;
    for(auto &p: particles) {
        Vec2 disp = pos - p.pos;
        density += smoothing_kernal(disp);
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

Vec2 compute_pressure_grad(Vec2 pos) {
    Vec2 grad = Vec2(0.0f, 0.0f);
    for(int i = 0; i < particles.size(); i++) {
        assert(densities[i] > 0);

        Vec2 kernel_grad = smoothing_kernal_grad(pos - particles[i].pos);
        grad = grad + kernel_grad * (pressures[i] / densities[i]);
    }
    return grad;
}

void compute_pressure_grads() {
    for(int i = 0; i < particles.size(); i++)
        pressure_grads[i] = compute_pressure_grad(particles[i].pos);
}

Vec2 compute_pressure_grad_particle(int index) {
    Vec2 grad = Vec2(0.0f, 0.0f);
    Vec2 pos = particles[index].pos;
    for(int i = 0; i < particles.size(); i++) {
        if(i == index)
            continue;
        assert(densities[i] > 0);
        float pressure = (pressures[i] + pressures[index]) * 0.5f;
        Vec2 kernel_grad = smoothing_kernal_grad(pos - particles[i].pos);
        grad = grad + kernel_grad * (pressure / densities[i]);
    }
    return grad;
}

void compute_pressure_grads_particle() {
    for(int i = 0; i < particles.size(); i++)
        pressure_grads[i] = compute_pressure_grad_particle(i);
}


Vec2 compute_density_grad(Vec2 pos) {
    Vec2 grad = Vec2(0, 0);
    for(auto &p: particles) {
        Vec2 disp = pos - p.pos;
        grad = grad + smoothing_kernal_grad(disp);
    }
    return grad;
}

void compute_density_grads() {
    for(int i = 0; i < particles.size(); i++)
        density_grads[i] = compute_density_grad(particles[i].pos);
}

GLuint textureID;  // OpenGL texture ID

void computePixelValue(int x, int y) {
    // This is a placeholder; replace it with your actual function
    // return static_cast<float>(x) / BOX_WIDTH + static_cast<float>(y) / BOX_HEIGHT;
    float density = compute_density(
        Vec2(
            (float)x / TEXTURE_SUBDIVS * BOX_WIDTH - BOX_WIDTH / 2, 
            (float)y / TEXTURE_SUBDIVS * BOX_HEIGHT - BOX_HEIGHT / 2
        )
    ) / max_density;
    glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, 1, 1, GL_RED, GL_FLOAT, &density);
}


void render_pressure(int x, int y) {
    // This is a placeholder; replace it with your actual function
    // return static_cast<float>(x) / BOX_WIDTH + static_cast<float>(y) / BOX_HEIGHT;
    Vec2 pos = Vec2(
        (float)x / TEXTURE_SUBDIVS * BOX_WIDTH - BOX_WIDTH / 2, 
        (float)y / TEXTURE_SUBDIVS * BOX_HEIGHT - BOX_HEIGHT / 2
    );
    float density = compute_density(pos);
    // printf("max density: %f, average density: %f\n", max_density, desired_density);
    // printf("%f \n", density);
    float pressure = compute_pressure(density);

    // float highest_pressure = compute_pressure(max_density);
    // printf("%f, %f, %f\n", desired_density, max_density, highest_pressure);
    float highest_pressure = 1000;

    // glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, 1, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, color);
    uint8_t color[] = {255, 255, 255, 255};
   
    float interp;
    if(pressure > 0) {
        interp = (uint8_t)(255 * (1 - pressure / highest_pressure));
        color[1] = interp;
        color[2] = interp;
    } else { 
        interp = (uint8_t)(255 * (1 - (desired_density - density) / desired_density));
        color[0] = interp;
        color[1] = interp;
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, 1, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, color);

    // if(density > desired_density) {
        // value = (density - desired_density) / desired_density;
        // printf("value: %f\n", value);
    // } else {
    //     value = (desired_density - density) / desired_density;
    //     glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, 1, 1, GL_BLUE, GL_FLOAT, &value);
    // }
    
}

void initializeTexture() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Create an empty texture with GL_RED format (grayscale)
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, TEXTURE_SUBDIVS, TEXTURE_SUBDIVS, 0, GL_RED, GL_FLOAT, nullptr);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEXTURE_SUBDIVS, TEXTURE_SUBDIVS, 0, GL_RGBA,  GL_UNSIGNED_INT_8_8_8_8, nullptr);
}

void updateTexture() {
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Update the texture data based on computePixelValue function
    for (int y = 0; y < TEXTURE_SUBDIVS; ++y) {
        for (int x = 0; x < TEXTURE_SUBDIVS; ++x) {
            // computePixelValue(x, y);
            render_pressure(x, y);
        }
    }
}

void drawTexturedQuad() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-BOX_WIDTH/2.0f, -BOX_HEIGHT/2.0f);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(BOX_WIDTH/2.0f, -BOX_HEIGHT/2.0f);

    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(BOX_WIDTH/2.0f, BOX_HEIGHT/2.0f);

    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-BOX_WIDTH/2.0f, BOX_HEIGHT/2.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

void draw_arrow(Vec2 start, Vec2 disp) {
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

inline void clamp_particle(Particle &p) {
    if(p.pos.x > BOX_WIDTH / 2) {
        p.pos.x = BOX_WIDTH - p.pos.x;
        p.pos.x = std::max(p.pos.x, -BOX_WIDTH / 2);
        p.vel.x = -std::abs(p.vel.x);
    } else if(p.pos.x < -BOX_WIDTH / 2) {
        p.pos.x = -BOX_WIDTH - p.pos.x;
        p.pos.x = std::min(p.pos.x, BOX_WIDTH / 2);
        p.vel.x = std::abs(p.vel.x);
    }

    if(p.pos.y > BOX_HEIGHT / 2) {
        p.pos.y = BOX_HEIGHT - p.pos.y;
        p.pos.y = std::max(p.pos.y, -BOX_HEIGHT / 2);
        p.vel.y = -std::abs(p.vel.y);
    } else if(p.pos.y < -BOX_HEIGHT / 2) {
        p.pos.y = -BOX_HEIGHT - p.pos.y;
        p.pos.y = std::min(p.pos.y, BOX_HEIGHT / 2);
        p.vel.y = std::abs(p.vel.y);
    }
}

void update_velocities() {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        
        Vec2 acc = pressure_grads[i] * (-1.0 / densities[i]);
        // Vec2 disp = p.vel * (0.5 * dt * dt) + p.vel * dt;
        
        Vec2 temp1 = p.vel * dt;
        Vec2 temp2 = acc * (dt * dt * 1); // 0.9 causes divergence, 1 leads to stability
        Vec2 disp = temp1 + temp2;
        p.vel = p.vel + acc * dt;
        // disp = disp + (acc * (0.5 * dt * dt));
        p.pos = p.pos + disp;
        clamp_particle(p);
        
        // p.vel = pressure_grads[i] * (-1.0 / densities[i]);
        // p.pos = p.pos + p.vel * dt / 10;
        // clamp_particle(p);
    }
}

void step_ahead() {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        particles_swap[i].pos = p.pos + p.vel * dt;
        particles_swap[i].vel = p.vel;
    }
}

void print_vec2(Vec2 v) {
    printf("(%f, %f)\n", v.x, v.y);
}

void print_particle(Particle &p) {
    printf("pos: (%f, %f), vel: (%f, %f)\n", p.pos.x, p.pos.y, p.vel.x, p.vel.y);
}

void report_time(Timer &t, const char *str) {
    printf("%s took %f seconds\n", str, t.time());
}

int main() {
    
    GLFWwindow *window = create_window();
    framebuffer_size_callback(window, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    tile_particles(particles);
    initializeTexture();

    float multiplier = 1.179;
    std::vector<Vec2> samples(SAMPLE_TILE_NUMBER * SAMPLE_TILE_NUMBER);
    for(int j = 0; j < SAMPLE_TILE_NUMBER; j++)
        for(int i = 0; i < SAMPLE_TILE_NUMBER; i++) {
            samples[j * SAMPLE_TILE_NUMBER + i] = Vec2(
                multiplier * (i - SAMPLE_TILE_NUMBER / 2.0f),
                multiplier * (j - SAMPLE_TILE_NUMBER / 2.0)
            );
        }

    while (!glfwWindowShouldClose(window)) {

        step_ahead();
        particles.swap(particles_swap);

        glClear(GL_COLOR_BUFFER_BIT);   
        glColor3f(1.0f, 1.0f, 1.0f);

        Timer time;
        compute_densities();
        report_time(time, "compute densities");

        time.reset();
        compute_pressures();
        report_time(time, "compute pressures");

        time.reset();
        compute_pressure_grads_particle();
        report_time(time, "compute pressure gradients");

        // time.reset();
        // updateTexture();
        // drawTexturedQuad();
        // report_time(time, "draw texture");

        time.reset();

        particles.swap(particles_swap);
        update_velocities();

        glColor3f(1.0f, 1.0f, 1.0f);
        // glColor3f(0.0f, 0.0f, 0.0f);
        for(auto &p: particles)
            renderCircle(p.pos.x, p.pos.y, PARTICLE_RADIUS);
        
        // printf("density at (0, 0): %f\n", compute_density(particles, Vec2(0, 0)));
        drawBox(-BOX_WIDTH / 2 + EPS, -BOX_WIDTH / 2 + EPS, BOX_HEIGHT / 2 - EPS, BOX_HEIGHT / 2 - EPS);

        // for(int i = 0; i < PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER; i++) {
        //     Vec2 sample = samples[i];

        //     glColor3f(0.0f, 1.0f, 0.0f);
        //     renderCircle(sample.x, sample.y, 0.1);

        //     draw_arrow(sample, compute_density_grad(sample));
        // }

        // for(int i = 0; i < SAMPLE_TILE_NUMBER * SAMPLE_TILE_NUMBER; i++) {
        //     Vec2 sample = samples[i];

        //     glColor3f(0.0f, 1.0f, 0.0f);
        //     renderCircle(sample.x, sample.y, 0.1);

        //     draw_arrow(sample, compute_pressure_grad(sample));
        // }

        // print_vec2(pressure_grads[0]);
        // print_particle(particles[0]);

        // renderCircle(7.200000, -7.300000, 0.05);
        // renderCircle(7.200000, -7.600000, 0.05);
        // print_particle(particles[19]);
        // // printf("%f, %f\n", compute_density(Vec2(7.2, -7.5)), compute_density(Vec2(7.5, -7.7)));
        // printf("%f, %f\n", 
        //     compute_pressure(compute_density(Vec2(7.200000, -7.300000))), 
        //     compute_pressure(compute_density(Vec2(7.200000, -7.600000)))
        // );
        // draw_arrow(Vec2(7.200000, -7.300000), compute_pressure_grad(Vec2(7.200000, -7.300000)));
        // draw_arrow(Vec2(7.200000, -7.600000), compute_pressure_grad(Vec2(7.200000, -7.600000)));

        report_time(time, "everything else");

        glfwSwapBuffers(window);
        glfwPollEvents();

        // std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}