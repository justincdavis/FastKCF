cmake_minimum_required(VERSION 3.10)
project(FastKCF)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno -ffinite-math-only") # clang

find_package(PkgConfig REQUIRED)     
pkg_check_modules(FFTW IMPORTED_TARGET REQUIRED fftw3)
          
if( NOT FFTW_ROOT AND DEFINED ENV{FFTWDIR} )
    set( FFTW_ROOT $ENV{FFTWDIR} )
endif()
          
find_library(
        FFTW_DOUBLE_THREADS_LIB
        NAMES "fftw3_threads"
        PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )
          
if (FFTW_DOUBLE_THREADS_LIB)
    set(FFTW_DOUBLE_THREADS_LIB_FOUND TRUE)
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_DOUBLE_THREADS_LIB})
    add_library(FFTW::DoubleThreads INTERFACE IMPORTED)
    set_target_properties(FFTW::DoubleThreads
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW_DOUBLE_THREADS_LIB}"
        )
else()
    set(FFTW_DOUBLE_THREADS_LIB_FOUND FALSE)
endif()
     
include_directories(PkgConfig::FFTW)
find_package(OpenGL REQUIRED)

# debugging
# add_link_options(-fsanitize=address)

add_compile_options(-ofast -fopenmp)
# add_compile_options(-ofast -fopenmp -funsafe-loop-optimizations -fipa-pta -fsched2-use-superblocks -mavx)
# add_compile_options(-o3 -fopenmp)
# add_compile_options(-L -lfftw3_omp -lfftw3 -lm)
add_compile_options(-g)

include_directories(kcfDetect PUBLIC
    extern/tracy/public/tracy
)

add_definitions(-DTRACY_ENABLE)
add_definitions(-DTRACY_NO_INVARIANT_CHECK=1 -DTRACY_TIMER_FALLBACK)

add_executable(kcfDetect main.cpp fastTracker.cpp fastTrackerMP.cpp fastTrackerCUDA.cpp fastTrackerMPCUDA.cpp extern/tracy/public/TracyClient.cpp utils.hpp)
target_link_libraries(kcfDetect PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${VTK_LIBRARIES} PkgConfig::FFTW ${FFTW_DOUBLE_THREADS_LIB} OpenGL::GL ${CMAKE_DL_LIBS})
