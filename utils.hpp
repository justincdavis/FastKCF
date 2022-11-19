#pragma once

#include "opencv2/core.hpp"
#include <iostream>

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

// /*
// *  multiply two dense matrices
// */
// inline void parallelMatrixMultiply(const cv::Mat src1, const cv::Mat src2, cv::Mat & dest) {
//     // perform a transpose on src1 and then run tiled matrix multiplication
//     std::cout << src1.rows << " " << src1.cols << " " << src2.rows << " " << src2.cols << std::endl;
//     cv::Mat t = src1 * src2;
//     std::cout << t.rows << " " << t.cols << std::endl;
//     dest.resize(src1.cols, src2.rows);
//     #pragma omp parallel for
//     for(int i = 0; i < src1.rows; i++) {
//         for(int k = 0; k < src2.cols; k++) {
//             float val = src1.at<float>(i, k);
//             for(int j = 0; j < src1.cols; j++) {
//                 dest.at<float>(i, j) = val * src2.at<float>(k, j);
//             }
//         }
//     }
// }
