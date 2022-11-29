#pragma once

#include "opencv2/core.hpp"

#include <memory>
#include <iostream>
#include <omp.h>
#include <fftw3.h>

#include "utils.hpp"

namespace cv{
    
//template <typename T>
//using Ptr = std::shared_ptr<T>; 

class FastTrackerMP
{
public:
    struct Params {
        float detect_thresh;         //!<  detection confidence threshold
        float sigma;                 //!<  gaussian kernel bandwidth
        float lambda;                //!<  regularization
        float interp_factor;         //!<  linear interpolation factor for adaptation
        float output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
        float pca_learning_rate;     //!<  compression learning rate
        bool resize;                  //!<  activate the resize feature to improve the processing speed
        bool split_coeff;             //!<  split the training coefficients into two matrices
        bool wrap_kernel;             //!<  wrap around the kernel values
        bool compress_feature;        //!<  activate the pca method to compress the features
        int max_patch_size;           //!<  threshold for the ROI size
        int compressed_size;          //!<  feature size after compression
        int desc_pca;        //!<  compressed descriptors of FastTrackerMP::MODE
        int desc_npca;       //!<  non-compressed descriptors of FastTrackerMP::MODE

        Params();
    };

    enum MODE {
      GRAY   = (1 << 0),
      CN     = (1 << 1),
      CUSTOM = (1 << 2)
    };

// IMPLEMENTATION DETAILS
private:
    bool failure();

    void createHanningWindow(OutputArray dest, const Size winSize, const int type) const;

    void inline fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const;
    void inline fft2(const Mat src, Mat & dest) const;
    void inline ifft2(const Mat src, Mat & dest) const;

    void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB=false) const;
    void inline sumChannels(std::vector<Mat> src, Mat & dest) const;
    void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix,float pca_rate, int compressed_sz,
                                       std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat v);
    void inline compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, FastTrackerMP::MODE desc = GRAY) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const;
    void extractCN(Mat patch_data, Mat & cnFeatures) const;
    void denseGaussKernel(const float sigma, const Mat , const Mat y_data, Mat & k_data,
                          std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf );
    void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const;
    void calcResponse(const Mat alphaf_data, const Mat alphaf_den_data, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const;

    void shiftRows(Mat& mat) const;
    void shiftRows(Mat& mat, int n) const;
    void shiftCols(Mat& mat, int n) const;

    float output_sigma;
    Rect2d roi;
    Mat hann; 	//hann window filter
    Mat hann_cn; //10 dimensional hann-window filter for CN features,

    Mat y,yf; 	// training response and its FFT
    Mat x; 	// observation and its FFT
    Mat k,kf;	// dense gaussian kernel and its FFT
    Mat kf_lambda; // kf+lambda
    Mat new_alphaf, alphaf;	// training coefficients
    Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
    Mat z; // model
    Mat response; // detection result
    Mat old_cov_mtx, proj_mtx; // for feature compression

    // pre-defined Mat variables for optimization of private functions
    Mat spec, spec2;
    std::vector<Mat> layers;
    std::vector<Mat> vxf,vyf,vxyf;
    Mat xy_data,xyf_data;
    Mat data_temp, compress_data;
    std::vector<Mat> layers_pca_data;
    std::vector<Scalar> average_data;
    Mat img_Patch;

    // storage for the extracted features, KRLS model, KRLS compressed model
    Mat X[2],Z[2],Zc[2];

    // storage of the extracted features
    std::vector<Mat> features_pca;
    std::vector<Mat> features_npca;
    std::vector<MODE> descriptors_pca;
    std::vector<MODE> descriptors_npca;

    // optimization variables for updateProjectionMatrix
    Mat data_pca, new_covar,w_data,u_data,vt_data;

    // custom feature extractor
    bool use_custom_extractor_pca;
    bool use_custom_extractor_npca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_pca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_npca;

    bool resizeImage; // resize the image whenever needed and the patch size is large

    int frame;

    // fftw stuff
    void inline write_fftw_image(const Mat src, fftw_complex * dest, const int height, const int width);
    void inline read_fftw_image(const fftw_complex * src, Mat & dest, Mat & t1, Mat & t2, const int height, const int width);

    // forward fft
    fftw_complex *f_in;
    fftw_complex *f_out;
    cv::Mat c1;
    cv::Mat c2;
    int f_h;
    int f_w;   
    fftw_plan f_plan;

    // backward fft
    fftw_complex *b_in;
    fftw_complex *b_out;
    fftw_plan b_plan;

    void inline fftw_fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data);
    void inline fftw_fft2(const Mat src, Mat & dest);
    void inline fftw_ifft2(const Mat src, Mat & dest);
///////

public:
    FastTrackerMP(FastTrackerMP::Params p);  // use ::create()
    
    Params params;

    static Ptr<FastTrackerMP> create() {
        FastTrackerMP::Params p = FastTrackerMP::Params{};
        return makePtr<FastTrackerMP>(p);
    }

    void init(InputArray image, const Rect& boundingBox);
    bool update(InputArray image, Rect& boundingBox);

    typedef void (*FeatureExtractorCallbackFN)(const Mat, const Rect, Mat&);
    void setFeatureExtractor(FeatureExtractorCallbackFN callback, bool pca_func = false);
};

} /* namespace cv */
