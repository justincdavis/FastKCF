# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jcdavis/FastKCF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jcdavis/FastKCF/build

# Include any dependencies generated for this target.
include CMakeFiles/kcfDetect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kcfDetect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kcfDetect.dir/flags.make

CMakeFiles/kcfDetect.dir/main.cpp.o: CMakeFiles/kcfDetect.dir/flags.make
CMakeFiles/kcfDetect.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kcfDetect.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kcfDetect.dir/main.cpp.o -c /home/jcdavis/FastKCF/main.cpp

CMakeFiles/kcfDetect.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kcfDetect.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcdavis/FastKCF/main.cpp > CMakeFiles/kcfDetect.dir/main.cpp.i

CMakeFiles/kcfDetect.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kcfDetect.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcdavis/FastKCF/main.cpp -o CMakeFiles/kcfDetect.dir/main.cpp.s

CMakeFiles/kcfDetect.dir/fastTracker.cpp.o: CMakeFiles/kcfDetect.dir/flags.make
CMakeFiles/kcfDetect.dir/fastTracker.cpp.o: ../fastTracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/kcfDetect.dir/fastTracker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kcfDetect.dir/fastTracker.cpp.o -c /home/jcdavis/FastKCF/fastTracker.cpp

CMakeFiles/kcfDetect.dir/fastTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kcfDetect.dir/fastTracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcdavis/FastKCF/fastTracker.cpp > CMakeFiles/kcfDetect.dir/fastTracker.cpp.i

CMakeFiles/kcfDetect.dir/fastTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kcfDetect.dir/fastTracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcdavis/FastKCF/fastTracker.cpp -o CMakeFiles/kcfDetect.dir/fastTracker.cpp.s

CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.o: CMakeFiles/kcfDetect.dir/flags.make
CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.o: ../fastTrackerMP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.o -c /home/jcdavis/FastKCF/fastTrackerMP.cpp

CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcdavis/FastKCF/fastTrackerMP.cpp > CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.i

CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcdavis/FastKCF/fastTrackerMP.cpp -o CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.s

CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.o: CMakeFiles/kcfDetect.dir/flags.make
CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.o: ../fastTrackerCUDA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.o -c /home/jcdavis/FastKCF/fastTrackerCUDA.cpp

CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcdavis/FastKCF/fastTrackerCUDA.cpp > CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.i

CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcdavis/FastKCF/fastTrackerCUDA.cpp -o CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.s

CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.o: CMakeFiles/kcfDetect.dir/flags.make
CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.o: ../fastTrackerMPCUDA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.o -c /home/jcdavis/FastKCF/fastTrackerMPCUDA.cpp

CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcdavis/FastKCF/fastTrackerMPCUDA.cpp > CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.i

CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcdavis/FastKCF/fastTrackerMPCUDA.cpp -o CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.s

CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.o: CMakeFiles/kcfDetect.dir/flags.make
CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.o: ../extern/tracy/public/TracyClient.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.o -c /home/jcdavis/FastKCF/extern/tracy/public/TracyClient.cpp

CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcdavis/FastKCF/extern/tracy/public/TracyClient.cpp > CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.i

CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcdavis/FastKCF/extern/tracy/public/TracyClient.cpp -o CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.s

# Object files for target kcfDetect
kcfDetect_OBJECTS = \
"CMakeFiles/kcfDetect.dir/main.cpp.o" \
"CMakeFiles/kcfDetect.dir/fastTracker.cpp.o" \
"CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.o" \
"CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.o" \
"CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.o" \
"CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.o"

# External object files for target kcfDetect
kcfDetect_EXTERNAL_OBJECTS =

kcfDetect: CMakeFiles/kcfDetect.dir/main.cpp.o
kcfDetect: CMakeFiles/kcfDetect.dir/fastTracker.cpp.o
kcfDetect: CMakeFiles/kcfDetect.dir/fastTrackerMP.cpp.o
kcfDetect: CMakeFiles/kcfDetect.dir/fastTrackerCUDA.cpp.o
kcfDetect: CMakeFiles/kcfDetect.dir/fastTrackerMPCUDA.cpp.o
kcfDetect: CMakeFiles/kcfDetect.dir/extern/tracy/public/TracyClient.cpp.o
kcfDetect: CMakeFiles/kcfDetect.dir/build.make
kcfDetect: /usr/local/lib/libopencv_gapi.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_stitching.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_alphamat.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_aruco.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_bgsegm.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_bioinspired.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_ccalib.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudabgsegm.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudafeatures2d.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudaobjdetect.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudastereo.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_dnn_superres.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_dpm.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_face.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_freetype.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_fuzzy.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_hdf.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_hfs.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_img_hash.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_intensity_transform.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_line_descriptor.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_mcc.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_quality.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_rapid.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_reg.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_rgbd.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_saliency.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_sfm.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_stereo.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_structured_light.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_superres.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_surface_matching.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_tracking.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_videostab.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_wechat_qrcode.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_xfeatures2d.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_xobjdetect.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_xphoto.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_shape.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_highgui.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_datasets.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_plot.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_text.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_ml.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_videoio.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudaoptflow.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudalegacy.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudawarping.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_optflow.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_ximgproc.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_video.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_dnn.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_imgcodecs.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_objdetect.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_calib3d.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_features2d.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_flann.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_photo.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudaimgproc.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudafilters.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_imgproc.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudaarithm.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_core.so.4.5.2
kcfDetect: /usr/local/lib/libopencv_cudev.so.4.5.2
kcfDetect: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
kcfDetect: /usr/lib/x86_64-linux-gnu/libpthread.so
kcfDetect: CMakeFiles/kcfDetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jcdavis/FastKCF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable kcfDetect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kcfDetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kcfDetect.dir/build: kcfDetect

.PHONY : CMakeFiles/kcfDetect.dir/build

CMakeFiles/kcfDetect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kcfDetect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kcfDetect.dir/clean

CMakeFiles/kcfDetect.dir/depend:
	cd /home/jcdavis/FastKCF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jcdavis/FastKCF /home/jcdavis/FastKCF /home/jcdavis/FastKCF/build /home/jcdavis/FastKCF/build /home/jcdavis/FastKCF/build/CMakeFiles/kcfDetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kcfDetect.dir/depend

