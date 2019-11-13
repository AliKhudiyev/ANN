cmake_minimum_required(VERSION 3.10)

project(ANN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES src/test.cpp src/ann.cpp src/layer.cpp src/perceptron.cpp src/dataset.cpp src/matrix.cpp)
# file(GLOB SOURCES src/*.cpp)

include_directories(include)

add_executable(main ${SOURCES})