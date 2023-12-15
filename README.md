sudo apt-get install libglfw3 libglfw3-dev
g++ main.cpp -o renderer $(pkg-config --cflags --libs glfw3)
a.k.a. g++ main.cpp -o renderer -lglfw

To build, do:
mkdir build
cd build
cmake ..
make

