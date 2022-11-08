#!/bin/bash
# clear OpenCV folder
cd ~
sudo rm -rf ~/OpenCV-cuDNN/

# TODO: download the .deb files????

mkdir ~/OpenCV-cuDNN
mkdir ~/OpenCV-cuDNN/OpenCV
mkdir ~/OpenCV-cuDNN/cuDNN
cd ~

# basic updates
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt update -y 
sudo apt upgrade -y

# opencv updates
sudo apt -y install build-essential \
    cmake \
    pkg-config \
    unzip \
    yasm \
    git \
    checkinstall \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \  
    libxvidcore-dev \
    x264 \
    libx264-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libfaac-dev \
    libmp3lame-dev \
    libvorbis-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libdc1394-22 \
    libdc1394-22-dev \
    libxine2-dev \
    libv4l-dev \
    v4l-utils

cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h -y
cd ~

sudo apt-get -y install libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-testresources \
    libtbb-dev \
    libatlas-base-dev \
    gfortran \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-dev \
    doxygen \
    ocl-icd-opencl-dev

sudo -H pip3 -y install -U pip numpy

# installs for CUDA
sudo apt-get -y install linux-headers-$(uname -r)
cd ~/Simulation/OpenCV-cuDNN/cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get -y update
sudo apt-get -y install cuda, nvidia-gds, zlib1g
cd ~

# opencv install and download
cd ~/OpenCV-cuDNN/OpenCV
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.2.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.2.zip
unzip opencv.zip
unzip opencv_contrib.zip
cd ~
