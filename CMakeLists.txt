cmake_minimum_required(VERSION 3.10)
project(FastKCF)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# optimization
#add_compile_options(-o3 -ffast-math)

# debugging
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

add_compile_options(-fopenmp)

add_executable(kcfDetect main.cpp fastTracker.cpp)
target_link_libraries(kcfDetect ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
