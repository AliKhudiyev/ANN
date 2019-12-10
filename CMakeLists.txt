cmake_minimum_required(VERSION 3.10)

project(ANN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC src/test.cpp)
set(LIBSRCS src/ann.cpp src/layer.cpp src/perceptron.cpp src/dataset.cpp src/matrix.cpp)
set(LIBDIR lib/)
# file(GLOB SOURCES src/*.cpp)

include_directories(include)
link_directories(lib)
add_library(ann STATIC ${LIBSRCS})

add_executable(main ${SRC})
target_link_libraries(main ann)
target_link_libraries(main argp)