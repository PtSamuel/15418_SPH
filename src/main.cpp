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

#define PARTICLES 10
#define PARTICLE_RADIUS 0.1f
#define PARTICLE_TILE_NUMBER 40
#define SAMPLE_TILE_NUMBER 10
#define OCCUPANCY 0.8f
#define BOX_WIDTH 20.0f
#define BOX_HEIGHT 40.0f
#define EPS 1e-3f
#define SMOOTH_RADIUS 1.5f
#define SMOOTH_RADIUS2 SMOOTH_RADIUS * SMOOTH_RADIUS
#define SMOOTH_RADIUS4 SMOOTH_RADIUS2 * SMOOTH_RADIUS2
#define TWO_THIRDS 2.0f / 3.0f

#define PRESSURE_RESPONSE 500.0f

#define TEXTURE_SUBDIVS 128

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

const float BLOCK_LEN = SMOOTH_RADIUS;
const int BLOCKS_X = static_cast<int>(std::ceil(BOX_WIDTH / BLOCK_LEN));
const int BLOCKS_Y = static_cast<int>(std::ceil(BOX_HEIGHT / BLOCK_LEN));

float momentum = 0.9;
float running_duration;

std::array<std::array<std::vector<Particle>, BLOCKS_Y>, BLOCKS_X> blocks;

// float SMOOTH_RADIUS8 = SMOOTH_RADIUS4 * SMOOTH_RADIUS4;
// float kernel_volume = M_PI / 4 * SMOOTH_RADIUS8;
// float normalizer = 1 / kernel_volume;

float kernel_volume = SMOOTH_RADIUS4 * M_PI / 6;
float normalizer = 1 / kernel_volume;

float average_density = PARTICLE_TILE_NUMBER * PARTICLE_TILE_NUMBER / (BOX_WIDTH * BOX_HEIGHT);
float desired_density = average_density * 2;

const float dt = 0.01;

static std::mt19937 gen(114514);
static std::uniform_real_distribution<float> distribution(0, 1);

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
            p.id = PARTICLE_TILE_NUMBER * j + i;
            p.pos.x = (float)(i - PARTICLE_TILE_NUMBER * 0.5) / PARTICLE_TILE_NUMBER * OCCUPANCY * BOX_WIDTH;
            p.pos.y = (float)(j - PARTICLE_TILE_NUMBER * 0.5) / PARTICLE_TILE_NUMBER * OCCUPANCY * BOX_HEIGHT;
        }
}

// float smoothing_kernal(Vec2 disp) {
//     float dist2 = disp.norm2();
//     float influence = std::max(0.0f, SMOOTH_RADIUS2 - dist2);
//     return influence * influence * influence * normalizer;
// }

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

void sanity_check_blocks() {
    int sum = 0;
    for(int i = 0; i < BLOCKS_X; i++) 
        for(int j = 0; j < BLOCKS_Y; j++) {
            sum += blocks[i][j].size();
            for(auto &p: blocks[i][j]) {
                auto coords = get_block(p.pos);
                assert(i == coords.first && j == coords.second);
            }
        }
    printf("num particle: %d\n", sum);
}

void report_block(Vec2 pos) {
    auto coords = get_block(pos);
    printf("(%f, %f), (%d, %d)\n", pos.x, pos.y, coords.first, coords.second);
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

    // std::set<int> set1;
    // for(int i = x - 1; i <= x + 1; i++)
    //     for(int j = y - 1; j <= y + 1; j++) {
    //         if(i < 0 || i >= BLOCKS_X || j < 0 || j >= BLOCKS_Y)
    //             continue;
    //         for(auto &p: blocks[i][j])
    //             set1.insert(p.id);
    //     }

    // std::set<int> set2;

    // float density = 0;

    // for(auto &p: particles) {
    //     Vec2 disp = pos - p.pos;
    //     density += smoothing_kernal(disp);
    //     auto temp = get_block(p.pos);
    //     if((std::abs(temp.first - x) <= 1) && (std::abs(temp.second - y) <= 1)) {
    //         set2.insert(p.id);
    //     }
    // }

    // assert(set1 == set2);
    // if(set1 != set2) {
    //     printf("frame: %d, %f, %f\n", frame, pos.x, pos.y);
    //     for(int i: set1) {
    //         printf("%d ", i);
    //     }
    //     printf("\n");

    //     for(int i: set2) {
    //         printf("%d ", i);
    //     }
    //     printf("\n");
        
    //     report_block(particles[25].pos);
        
    //     // printf("%d, %d, %d, %d\n", temp.first, temp.second, x, y);
        
    //     exit(1);
    // }

    // int dummy;
    // scanf("%d", &dummy);

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

    auto coords = get_block(pos);
    int x = coords.first, y = coords.second;
    
    Vec2 grad = Vec2(0.0f, 0.0f);
    for(int i = 0; i < particles.size(); i++) {
        assert(densities[i] > 0);

        Vec2 kernel_grad = smoothing_kernal_grad(pos - particles[i].pos);
        grad = grad + kernel_grad * (pressures[i] / densities[i]);
    }

    // for(int i = x - 1; i <= x + 1; i++)
    //     for(int j = y - 1; j <= y + 1; j++) {
    //         if(i < 0 || i >= BLOCKS_X || j < 0 || j >= BLOCKS_Y)
    //             continue;
            
    //         for(auto &p: blocks[i][j]) {
    //             assert(densities[p.id] > 0);
    //             Vec2 kernel_grad = smoothing_kernal_grad(pos - p.pos);
    //             grad = grad + kernel_grad * (pressures[p.id] / densities[p.id]);
    //         }
    //     }

    return grad;
}

void compute_pressure_grads() {
    for(int i = 0; i < particles.size(); i++)
        pressure_grads[i] = compute_pressure_grad(particles[i].pos);
}

Vec2 compute_pressure_grad_particle(int index) {
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

// Vec2 compute_pressure_grad_particle(int index) {
//     Vec2 grad1 = Vec2(0.0f, 0.0f);
//     Vec2 pos = particles[index].pos;

//     std::set<int> set1, set2;

//     auto coords = get_block(pos);
//     int x = coords.first, y = coords.second;
//     for(int i = x - 1; i <= x + 1; i++)
//         for(int j = y - 1; j <= y + 1; j++) {
//             if(i < 0 || i >= BLOCKS_X || j < 0 || j >= BLOCKS_Y)
//                 continue;
            
//             for(auto &p: blocks[i][j]) {
//                 if(p.id == index)
//                     continue;
//                 assert(densities[p.id] > 0);

//                 float pressure = (pressures[p.id] + pressures[index]) * 0.5f;
//                 set1.insert(p.id);

//                 Vec2 kernel_grad = smoothing_kernal_grad(pos - p.pos);
//                 grad1 = grad1 + kernel_grad * (pressure / densities[p.id]);
//             }
//         }

//     // return grad;

//     Vec2 grad2 = Vec2(0.0f, 0.0f);
//     for(int i = 0; i < particles.size(); i++) {
//         if(i == index)
//             continue;
//         assert(densities[i] > 0);
//         float pressure = (pressures[i] + pressures[index]) * 0.5f;
        
//         auto temp = get_block(particles[i].pos);
//         if(std::abs(temp.first - x) <= 1 && std::abs(temp.second - y) <= 1)
//             set2.insert(i);

//         Vec2 kernel_grad = smoothing_kernal_grad(pos - particles[i].pos);
//         grad2 = grad2 + kernel_grad * (pressure / densities[i]);
//     }

//     if(set1 != set2) {
//         for(int i: set1) {
//             printf("%d ", i);
//         }
//         printf("\n");

//         for(int i: set2) {
//             printf("%d ", i);
//         }
//         printf("\n");
//         exit(1);
//     }
//     assert((grad1 - grad2).norm2() < 1e-4);

//     return grad1;
// }

void compute_pressure_grads_particle() {
    for(int i = 0; i < particles.size(); i++)
        pressure_grads[i] = compute_pressure_grad_particle(i);
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
        
        p.vel = p.vel + x_dots[i].acc * dt;
        p.pos = p.pos + p.vel * dt;

        clamp_particle(p);
    }
}

inline Vec2 compute_acc(int index) {
    return pressure_grads[index] * (-1.0 / densities[index]) + Vec2(0.0f, -9.8f);
    // return pressure_grads[index] * (-1.0 / densities[index]);
}

void step_ahead() {
    for(int i = 0; i < particles.size(); i++) {
        Particle &p = particles[i];
        particles_swap[i].pos = p.pos + p.vel * dt;
        // particles_swap[i].vel = p.vel + x_dots[i].acc * dt * TWO_THIRDS;
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
        // x_dots[i].vel = x_dots[i].vel;
        x_dots[i].acc = s.acc;
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

    printf("BLOCK_X: %d, BLOCK_Y: %d\n", BLOCKS_X, BLOCKS_Y);
    
    GLFWwindow *window = create_window();
    framebuffer_size_callback(window, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    tile_particles(particles);
    particles_swap = particles;

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

    
    Timer duration;

    while (!glfwWindowShouldClose(window)) {

        duration.reset();

        frame++;

        distribute();
        // sanity_check_blocks();

        glClear(GL_COLOR_BUFFER_BIT);   
        glColor3f(1.0f, 1.0f, 1.0f);

        Timer time;
        compute_densities();
        // report_time(time, "compute densities");

        time.reset();
        compute_pressures();
        // report_time(time, "compute pressures");

        time.reset();
        compute_pressure_grads_particle();
        // report_time(time, "compute pressure gradients");

        // time.reset();
        // updateTexture();
        // drawTexturedQuad();
        // report_time(time, "draw texture");

        time.reset();

        compute_x_dot();

        step_ahead();
        particles.swap(particles_swap);
        distribute();

        compute_densities();
        compute_pressures();
        compute_pressure_grads_particle();

        increment_x_dot(0.75);
        particles.swap(particles_swap);

        update_velocities();

        glColor3f(1.0f, 1.0f, 1.0f);
        for(auto &p: particles)
            renderCircle(p.pos.x, p.pos.y, PARTICLE_RADIUS);
        
        drawBox(-BOX_WIDTH / 2 + EPS, -BOX_HEIGHT / 2 + EPS, BOX_WIDTH / 2 - EPS, BOX_HEIGHT / 2 - EPS);

        // for(int i = 0; i < SAMPLE_TILE_NUMBER * SAMPLE_TILE_NUMBER; i++) {
        //     Vec2 sample = samples[i];

        //     glColor3f(0.0f, 1.0f, 0.0f);
        //     renderCircle(sample.x, sample.y, 0.1);

        //     draw_arrow(sample, compute_pressure_grad(sample));
        // }

        // report_time(time, "everything else");

        if(frame == 200) {
            print_particle(particles[0]);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        running_duration = momentum * running_duration + (1 - momentum) * duration.time();

        if(frame % 100 == 0) {
            printf("fps: %f\n", 1 / running_duration);
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}