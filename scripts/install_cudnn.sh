#!/bin/bash
# sudo dpkg -i ./cuDNN/libcudnn8_8.2.0.53-1+cuda11.3_amd64.deb
# sudo dpkg -i ./cuDNN/libcudnn8-dev_8.2.0.53-1+cuda11.3_amd64.deb
# sudo dpkg -i ./cuDNN/libcudnn8-samples_8.2.0.53-1+cuda11.3_amd64.deb
sudo apt-get -y update
sudo apt-get -y install libcudnn8=8.2.0.53-1+cuda11.3 \
    libcudnn8-dev=8.2.0.53-1+cuda11.3 \
    libcudnn8-samples=8.2.0.53-1+cuda11.3 \
    g++ \
    freeglut3-dev \
    build-essential \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libfreeimage3 \
    libfreeimage-dev
