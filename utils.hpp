#pragma once

#include "opencv2/core.hpp"
#include <iostream>

#include <fftw3.h>

/*
*  perform elementwise multiplication of a matrix with a scalar
*/
inline void parallelElementWiseMult(cv::Mat & src, const float scalar, const int batch_size) {
    const int area = src.rows * src.cols;
    #pragma omp parallel for
    for (int idx = 0; idx < area; idx+=batch_size) {
        for (int offset = 0; offset < batch_size; offset++) {
            int h = idx + offset;
            if (h > area) break;
            int i = h / src.cols;
            int j = h % src.cols;
            src.at<float>(i, j) *= scalar;
        }
    }
}

inline void parallelElementWiseMult(cv::Mat & src, cv::Mat & dest, const float scalar, const int batch_size) {
    const int area = src.rows * src.cols;
    #pragma omp parallel for
    for (int idx = 0; idx < area; idx+=batch_size) {
        for (int offset = 0; offset < batch_size; offset++) {
            int h = idx + offset;
            if (h > area) break;
            int i = h / src.cols;
            int j = h % src.cols;
            src.at<float>(i, j) *= scalar;
        }
    }
    dest = src.clone();
}

inline void createFFTW3Image(cv::Mat & src, fftw_complex * & dest, const int height, const int width) {
    int k = 0;
    #pragma omp parallel for private(k)
    for(int j = 0 ; j < height ; j++ ) {
        for(int i = 0 ; i < width ; i++ ) {
            dest[k][0] = ( double )src.at<float>(j, i);
            dest[k][1] = 0.0;
            k++;
        }
    }
}

inline void readFFTW3Image(fftw_complex * & src, cv::Mat & dest, const int height, const int width){
    // normalize
    const double c = (double)(height * width);
    for(int i = 0 ; i < dest.rows * dest.cols ; i++ ) {
        src[i][0] /= c;
    }

    // copy
    int k = 0;
    #pragma omp parallel for private(k)
    for(int j = 0 ; j < dest.rows ; j++ ) {
        for(int i = 0 ; i < dest.cols ; i++ ) {
            dest.at<float>(j, i) = src[k][0];
            k++;
        }
    }
}
