#include <iostream>
#include <GLFW/glfw3.h>

int main() {
    // GLFW initialization code here

    if (!glfwInit())
    {
        std::cout << "failed" << std::endl;
    }
    // if (!glfwInit()) {
    //     std::cerr << "Failed to initialize GLFW" << std::endl;
    //     return -1;
    // }

    // // GLFW window creation and event loop here

    // // Clean up GLFW
    // glfwTerminate();

    return 0;
}