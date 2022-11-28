cmake_minimum_required(VERSION 3.10)
project(FastKCF)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# debugging
# add_link_options(-fsanitize=address)

add_compile_options(-ofast -fopenmp -funsafe-loop-optimizations -fipa-pta -fsched2-use-superblocks -mavx -march=skylake)
add_compile_options(-g)

include_directories(kcfDetect PUBLIC
    extern/tracy/public/tracy
)

add_definitions(-DTRACY_ENABLE)
add_definitions(-DTRACY_NO_INVARIANT_CHECK=1 -DTRACY_TIMER_FALLBACK)

add_executable(kcfDetect main.cpp fastTracker.cpp fastTrackerMP.cpp fastTrackerCUDA.cpp fastTrackerMPCUDA.cpp extern/tracy/public/TracyClient.cpp utils.hpp)
target_link_libraries(kcfDetect ${OpenCV_LIBS} OpenMP::OpenMP_CXX dl)
