#include "fastTrackerFFTW.hpp"
#include <complex>
#include <cmath>
#include "colorNames.hpp"

#include <opencv2/imgproc.hpp> // INTER_LINEAR_EXACT

#include "Tracy.hpp"

namespace cv {
//inline namespace tracking {
//namespace impl {


//template <typename T>
//Ptr<T> makePtr() {
//    return std::make_shared<T>();
//}

  FastTrackerFFTW::FastTrackerFFTW(FastTrackerFFTW::Params p) :
    params(p)
  {
    resizeImage = false;
    use_custom_extractor_pca = false;
    use_custom_extractor_npca = false;
  }

  bool FastTrackerFFTW::failure() {
    failed = true;
    fftw_cleanup_threads();
    fftw_cleanup();
    return false;
  }

  // /*
  // *  perform elementwise multiplication of a matrix with a scalar
  // */
  // void FastTrackerFFTW::parallelElementWiseMult(Mat & src, const float scalar, const int batch_size) {
  //   const int area = src.rows * src.cols;
  //   #pragma omp parallel for
  //   for (int idx = 0; idx < area; idx+=batch_size) {
  //     for (int offset = 0; offset < batch_size; offset++) {
  //       int h = idx + offset;
  //       if (h > area) break;
  //       int i = h / src.cols;
  //       int j = h % src.cols;
  //       src.at<float>(i, j) *= scalar;
  //     }
  //   }
  // }

  /*
   * Initialization:
   * - creating hann window filter
   * - ROI padding
   * - creating a gaussian response for the training ground-truth
   * - perform FFT to the gaussian response
   */
  void FastTrackerFFTW::init(InputArray image, const Rect& boundingBox)
  {
    ZoneScopedN("ftmp init");

    frame=0;
    roi.x = cvRound(boundingBox.x);
    roi.y = cvRound(boundingBox.y);
    roi.width = cvRound(boundingBox.width);
    roi.height = cvRound(boundingBox.height);

    // // print out the roi dimensions
    // std::cout << "ROI HEIGHT: " << roi.height << ", ROI WIDTH: " << roi.width << std::endl;

    // enable fftw threads
    if (!failed) {
      failed = failure();
    }
    failed = false;

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    // import wisdom
    fftw_import_system_wisdom();
    fftw_import_wisdom_from_filename("wisdom");

    // generate fftw plan
    f_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * roi.width * roi.height);
    f_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * roi.width * roi.height);
    c1 = cv::Mat::zeros(roi.height, roi.width, CV_32FC1);
    c2 = cv::Mat::zeros(roi.height, roi.width, CV_32FC1);
    f_h = roi.height;
    f_w = roi.width;
    f_plan = fftw_plan_dft_2d(f_h, f_w, f_in, f_out, FFTW_FORWARD, FFTW_MEASURE);

    //calclulate output sigma
    output_sigma=std::sqrt(static_cast<float>(roi.width*roi.height))*params.output_sigma_factor;
    output_sigma=-0.5f/(output_sigma*output_sigma);

    //resize the ROI whenever needed
    if(params.resize && roi.width*roi.height>params.max_patch_size){
      resizeImage=true;
      roi.x/=2.0;
      roi.y/=2.0;
      roi.width/=2.0;
      roi.height/=2.0;
    }

    // add padding to the roi
    roi.x-=roi.width/2;
    roi.y-=roi.height/2;
    roi.width*=2;
    roi.height*=2;

    //std::cout << "Creating hann window" << std::endl;
    // initialize the hann window filter
    createHanningWindow(hann, roi.size(), CV_32F);

    //std::cout << "Running merge for 10 hann windows" << std::endl;
    // hann window filter for CN feature
    Mat _layer[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
    merge(_layer, 10, hann_cn);

    //std::cout << "Creating the gaussian response" << std::endl;
    // create gaussian response
    y=Mat::zeros((int)roi.height,(int)roi.width,CV_32F);

    const float half_height = roi.height/2;
    const float half_width = roi.width/2;
    #pragma omp parallel for
    for(int i=0;i<int(roi.height);i++){
      for(int j=0;j<int(roi.width);j++){
        y.at<float>(i,j) =
                static_cast<float>((i-half_height+1)*(i-half_height+1)+(j-half_width+1)*(j-half_width+1));
      }
    }

    //std::cout << "Run cv::exp" << std::endl;
    //y*=(float)output_sigma;
    
    // can move into above loop?
    parallelElementWiseMult(y, output_sigma, 10);

    cv::exp(y,y);

    //std::cout << "Perform fft2" << std::endl;
    // perform fourier transfor to the gaussian response

    fftw_fft2(y,yf);
    // fft2(y, yf);

    //std::cout << "Disable ColorNames for grayscale images" << std::endl;
    if (image.channels() == 1) { // disable CN for grayscale images
      params.desc_pca &= ~(CN);
      params.desc_npca &= ~(CN);
    }
    //model = makePtr<FastTrackerFFTWModel>();

    //std::cout << "Record the non-compressed descriptors" << std::endl;
    // record the non-compressed descriptors
    if((params.desc_npca & GRAY) == GRAY)descriptors_npca.push_back(GRAY);
    if((params.desc_npca & CN) == CN)descriptors_npca.push_back(CN);
    if(use_custom_extractor_npca)descriptors_npca.push_back(CUSTOM);
    features_npca.resize(descriptors_npca.size());

    //std::cout << "Record the compressed descriptors" << std::endl;
    // record the compressed descriptors
    if((params.desc_pca & GRAY) == GRAY)descriptors_pca.push_back(GRAY);
    if((params.desc_pca & CN) == CN)descriptors_pca.push_back(CN);
    if(use_custom_extractor_pca)descriptors_pca.push_back(CUSTOM);
    features_pca.resize(descriptors_pca.size());

    // accept only the available descriptor modes
    CV_Assert(
      (params.desc_pca & GRAY) == GRAY
      || (params.desc_npca & GRAY) == GRAY
      || (params.desc_pca & CN) == CN
      || (params.desc_npca & CN) == CN
      || use_custom_extractor_pca
      || use_custom_extractor_npca
    );

    //std::cout << "Ensure the roi has intersection with the image" << std::endl;
    // ensure roi has intersection with the image
    Rect2d image_roi(0, 0,
                     image.cols() / (resizeImage ? 2 : 1),
                     image.rows() / (resizeImage ? 2 : 1));
    CV_Assert(!(roi & image_roi).empty());
  }

  /*
   * Main part of the KCF algorithm
   */
  bool FastTrackerFFTW::update(InputArray image, Rect& boundingBoxResult)
  {
    ZoneScopedN("ftmp update");
    double minVal, maxVal;	// min-max response
    Point minLoc,maxLoc;	// min-max location

    CV_Assert(image.channels() == 1 || image.channels() == 3);

    Mat img;
    {
      ZoneScopedN("resize");
    // resize the image whenever needed
    if (resizeImage)
        resize(image, img, Size(image.cols()/2, image.rows()/2), 0, 0, INTER_LINEAR_EXACT);
    else
        image.copyTo(img);
    }

    // detection part
    if(frame>0){

      // TODO - we can switch this to run the subwindow extraction in parallel
      // extract and pre-process the patch
      // get non compressed descriptors
      //#pragma omp parallel for
      for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
        if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i])) return failure();
      }
      //get non-compressed custom descriptors
      unsigned j = (unsigned)(descriptors_npca.size()-extractor_npca.size());
      //#pragma omp parallel for private(j)
      for(unsigned i=0;i<extractor_npca.size();i++){
        j++;
        if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i])) return failure();
      }

      if(features_npca.size()>0) {
        ZoneScopedN("merge");
        merge(features_npca,X[1]);
      }

      // get compressed descriptors
      for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
        if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i])) return failure();
      }
      //get compressed custom descriptors
      for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
        if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i])) return failure();
      }

      if(features_pca.size()>0) {
        ZoneScopedN("merge");
        merge(features_pca,X[0]);
      }

      //compress the features and the KRSL model
      if(params.desc_pca !=0){
        ZoneScopedN("compress");
        compress(proj_mtx,X[0],X[0],data_temp,compress_data);
        compress(proj_mtx,Z[0],Zc[0],data_temp,compress_data);
      }

      // copy the compressed KRLS model
      Zc[1] = Z[1];

      {
        ZoneScopedN("feature merge");
      // merge all features
      if(features_npca.size()==0){
        x = X[0];
        z = Zc[0];
      }else if(features_pca.size()==0){
        x = X[1];
        z = Z[1];
      }else{
        merge(X,2,x);
        merge(Zc,2,z);
      }
      }

      //compute the gaussian kernel
      denseGaussKernel(params.sigma,x,z,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

      // compute the fourier transform of the kernel
      // fft2(k, kf);
      fftw_fft2(k,kf);
      if(frame==1)spec2=Mat_<Vec2f >(kf.rows, kf.cols);

      // calculate filter response
      if(params.split_coeff)
        calcResponse(alphaf,alphaf_den,kf,response, spec, spec2);
      else
        calcResponse(alphaf,kf,response, spec);

      // extract the maximum response
      minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
      if (maxVal < params.detect_thresh)
      {
          return failure();
      }
      roi.x+=(maxLoc.x-roi.width/2+1);
      roi.y+=(maxLoc.y-roi.height/2+1);
    }

    // update the bounding box
    Rect2d boundingBox;
    boundingBox.x=(resizeImage?roi.x*2:roi.x)+(resizeImage?roi.width*2:roi.width)/4;
    boundingBox.y=(resizeImage?roi.y*2:roi.y)+(resizeImage?roi.height*2:roi.height)/4;
    boundingBox.width = (resizeImage?roi.width*2:roi.width)/2;
    boundingBox.height = (resizeImage?roi.height*2:roi.height)/2;

    // TODO - we can switch this to run the subwindow extraction in parallel
    // extract the patch for learning purpose
    // get non compressed descriptors
    for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
      if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i])) return failure();
    }
    //get non-compressed custom descriptors
    for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
      if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i])) return failure();
    }
    if(features_npca.size()>0)merge(features_npca,X[1]);

    // get compressed descriptors
    for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
      if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i])) return failure();
    }
    //get compressed custom descriptors
    for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
      if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i])) return failure();
    }
    if(features_pca.size()>0)merge(features_pca,X[0]);

    //update the training data
    if(frame==0){
      Z[0] = X[0].clone();
      Z[1] = X[1].clone();
    }else{
      Z[0]=(1.0-params.interp_factor)*Z[0]+params.interp_factor*X[0];
      Z[1]=(1.0-params.interp_factor)*Z[1]+params.interp_factor*X[1];
    }

    if(params.desc_pca !=0 || use_custom_extractor_pca){
      // initialize the vector of Mat variables
      if(frame==0){
        layers_pca_data.resize(Z[0].channels());
        average_data.resize(Z[0].channels());
      }

      // feature compression
      updateProjectionMatrix(Z[0],old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size,layers_pca_data,average_data,data_pca, new_covar,w_data,u_data,vt_data);
      compress(proj_mtx,X[0],X[0],data_temp,compress_data);
    }

    {
      ZoneScopedN("merge features");
    // merge all features
    if(features_npca.size()==0)
      x = X[0];
    else if(features_pca.size()==0)
      x = X[1];
    else
      merge(X,2,x);
    }

    // initialize some required Mat variables
    if(frame==0){
      layers.resize(x.channels());
      vxf.resize(x.channels());
      vyf.resize(x.channels());
      vxyf.resize(vyf.size());
      new_alphaf=Mat_<Vec2f >(yf.rows, yf.cols);
    }

    // Kernel Regularized Least-Squares, calculate alphas
    denseGaussKernel(params.sigma,x,x,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

    // compute the fourier transform of the kernel and add a small value
    // fft2(k,kf);
    fftw_fft2(k,kf);
    kf_lambda=kf+params.lambda;

    {
      ZoneScopedN("post-fft stuff");
    float den;
    if(params.split_coeff){
      mulSpectrums(yf,kf,new_alphaf,0);
      mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
    }else{
       
      for(int i=0;i<yf.rows;i++){
        for(int j=0;j<yf.cols;j++){
          den = 1.0f/(kf_lambda.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[0]+kf_lambda.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[1]);

          new_alphaf.at<Vec2f>(i,j)[0]=
          (yf.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[0]+yf.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[1])*den;
          new_alphaf.at<Vec2f>(i,j)[1]=
          (yf.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[0]-yf.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[1])*den;
        }
      }
    }
    }

    {
      ZoneScopedN("update RLS");
    // update the RLS model
    if(frame==0){
      alphaf=new_alphaf.clone();
      if(params.split_coeff)alphaf_den=new_alphaf_den.clone();
    }else{
      alphaf=(1.0-params.interp_factor)*alphaf+params.interp_factor*new_alphaf;
      if(params.split_coeff)alphaf_den=(1.0-params.interp_factor)*alphaf_den+params.interp_factor*new_alphaf_den;
    }
    }

    frame++;

    int x1 = cvRound(boundingBox.x);
    int y1 = cvRound(boundingBox.y);
    int x2 = cvRound(boundingBox.x + boundingBox.width);
    int y2 = cvRound(boundingBox.y + boundingBox.height);
    boundingBoxResult = Rect(x1, y1, x2 - x1, y2 - y1) & Rect(Point(0, 0), image.size());

    return true;
  }


  /*-------------------------------------
  |  implementation of the KCF functions
  |-------------------------------------*/

  /*
   * hann window filter
   */
  void FastTrackerFFTW::createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const {
      ZoneScopedN("ftmp hanningWindow");
      CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

      dest.create(winSize, type);
      Mat dst = dest.getMat();

      int rows = dst.rows, cols = dst.cols;

      AutoBuffer<float> _wc(cols);
      float * const wc = _wc.data();

      const float coeff0 = 2.0f * (float)CV_PI / (cols - 1);
      const float coeff1 = 2.0f * (float)CV_PI / (rows - 1);
       
      for(int j = 0; j < cols; j++)
        wc[j] = 0.5f * (1.0f - cos(coeff0 * j));

      if(dst.depth() == CV_32F){
         
        for(int i = 0; i < rows; i++){
          float* dstData = dst.ptr<float>(i);
          float wr = 0.5f * (1.0f - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = (float)(wr * wc[j]);
        }
      }else{
         
        for(int i = 0; i < rows; i++){
          double* dstData = dst.ptr<double>(i);
          double wr = 0.5f * (1.0f - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = wr * wc[j];
        }
      }

      // perform batch sqrt for SSE performance gains
      //cv::sqrt(dst, dst); //matlab do not use the square rooted version
  }

  /*
   * methods for reading and writing images to fftw format
   */
  void inline FastTrackerFFTW::write_fftw_image(const Mat src, fftw_complex * dest, const int height, const int width) {
    ZoneScopedN("write_fftw_image");
    // #pragma omp parallel for
    // for(int j = 0 ; j < height ; j++ ) {
    //   int k = j*width;
    //   for(int i = 0 ; i < width ; i++ ) {
    //     int l = k + i;
    //     dest[l][0] = ( double )src.at<float>(j, i);
    //     dest[l][1] = 0.0;
    //   }
    // }
    const int area = height * width;
    const int batch_size = 10;
    #pragma omp parallel for
    for(int idx = 0; idx < area; idx += batch_size){
      for (int offset = 0; offset < batch_size; offset++) {
            int h = idx + offset;
            if (h > area) break;
            int i = h / width;
            int j = h % width;

            dest[h][0] = ( double )src.at<float>(i, j);
            dest[h][1] = 0.0;
        }
    }
  }

  // TODO, I actually do want to pass these by reference
  void inline FastTrackerFFTW::read_fftw_image(const fftw_complex * src, Mat & dest, Mat & t1, Mat & t2, const int height, const int width) {
    ZoneScopedN("read_fftw_image");
    // normalize
    const double c = (double)(height * width);
    // for(int i = 0 ; i < dest.rows * dest.cols ; i++ ) {
    //     src[i][0] /= c;
    // }
    // copy
    // int k = 0;
    // // #pragma omp parallel for shared(k)
    // for(int j = 0 ; j < dest.rows ; j++ ) {
    //     for(int i = 0 ; i < dest.cols ; i++ ) {
    //         t1.at<float>(j, i) = src[k][0] / c;
    //         k++;
    //     }
    // }
    // k = 0;
    // // #pragma omp parallel for shared(k)
    // for(int j = 0 ; j < dest.rows ; j++ ) {
    //     for(int i = 0 ; i < dest.cols ; i++ ) {
    //         t2.at<float>(j, i) = src[k][1] / c;
    //         k++;
    //     }
    // }
    
    // TODO can be much better
    // this optimization reduced time by 10ms
    const int area = height * width;
    const int batch_size = 20;
    #pragma omp parallel for
    for(int idx = 0; idx < area; idx += batch_size){
      int is[batch_size];
      int js[batch_size];
      int top = 0;
      for (int offset = 0; offset < batch_size; offset++) {
            int h = idx + offset;
            if (h > area) break;
            is[offset] = h / width;
            js[offset] = h % width;
            top = offset;
      }
      for (int x = 0; x < batch_size; x++) {
        if (x > top) break;
        t1.at<float>(is[x], js[x]) = src[idx+x][0] / c;
      }
      for (int x = 0; x < batch_size; x++) {
        if (x > top) break;
        t2.at<float>(is[x], js[x]) = src[idx+x][1] / c;
      }
    }

    cv::merge(std::vector<cv::Mat>{t1, t2}, dest);
  }

  /*
   * simplification of fourier transform function in opencv
   */
  void inline FastTrackerFFTW::fft2(const Mat src, Mat & dest) const {
    ZoneScopedN("ftmp fft2");
    dft(src,dest,DFT_COMPLEX_OUTPUT);
  }

  void inline FastTrackerFFTW::fftw_fft2(const Mat src, Mat & dest) {
    ZoneScopedN("ftmp fftw_fft2");
    write_fftw_image(src, f_in, f_h, f_w);
    fftw_execute(f_plan);
    read_fftw_image(f_out, dest, c1, c2, f_h, f_w);
  }

  void inline FastTrackerFFTW::fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const {
    ZoneScopedN("ftmp fft2");
    split(src, layers_data);

    #pragma omp parallel for
    for(int i=0;i<src.channels();i++){
      dft(layers_data[i],dest[i],DFT_COMPLEX_OUTPUT);
    }
  }

  void inline FastTrackerFFTW::fftw_fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) {
    ZoneScopedN("ftmp fftw_fft2");
    split(src, layers_data);

    for(int i=0;i<src.channels();i++){
      write_fftw_image(layers_data[i], f_in, f_h, f_w);
      fftw_execute(f_plan);
      read_fftw_image(f_out, dest[i], c1, c2, f_h, f_w);
    }
  }

  /*
   * simplification of inverse fourier transform function in opencv
   */
  void inline FastTrackerFFTW::ifft2(const Mat src, Mat & dest) const {
    ZoneScopedN("ftmp ifft2");
    idft(src,dest,DFT_SCALE+DFT_REAL_OUTPUT);
  }

  void inline FastTrackerFFTW::fftw_ifft2(const Mat src, Mat & dest) {
    // TODO
  }

  /*
   * Point-wise multiplication of two Multichannel Mat data
   */
  void inline FastTrackerFFTW::pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB) const {
    ZoneScopedN("ftmp pxWiseMult");
    for(unsigned i=0;i<src1.size();i++){
      mulSpectrums(src1[i], src2[i], dest[i],flags,conjB);
    }
  }

  /*
   * Combines all channels in a multi-channels Mat data into a single channel
   */
  void inline FastTrackerFFTW::sumChannels(std::vector<Mat> src, Mat & dest) const {
    ZoneScopedN("ftmp sumChans");
    dest=src[0].clone();
    
    for(unsigned i=1;i<src.size();i++){
      dest+=src[i];
    }
  }

  /*
   * obtains the projection matrix using PCA
   */
  void inline FastTrackerFFTW::updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix, float pca_rate, int compressed_sz,
                                                     std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat vt) {
    ZoneScopedN("ftmp upProjMat");
    CV_Assert(compressed_sz<=src.channels());

    {
      ZoneScopedN("split");
    split(src,layers_pca);
    }

    {
      ZoneScopedN("avg");
    #pragma omp parallel for
    for (int i=0;i<src.channels();i++){
      average[i]=mean(layers_pca[i]);
      layers_pca[i]-=average[i];
    }
    }

    {
      ZoneScopedN("covar");
    // calc covariance matrix
    merge(layers_pca,pca_data);
    pca_data=pca_data.reshape(1,src.rows*src.cols);
    }
    
    // this is the cursed line
    // TODO OPTIMIZE
    {
      ZoneScopedN("new_conv");
      // take the transpose of a matrix and multiple it by itself
      auto t = pca_data.t();
      
      new_cov = t * pca_data;
      // cv::Mat new_cov;
      // parallelMatrixMultiply(t, pca_data, new_cov);

      const float scale = 1.0 / (float)(src.rows * src.cols-1);
      const int batch_size = 10;

      parallelElementWiseMult(new_cov, scale, batch_size);

      // new_cov=1.0/(float)(src.rows*src.cols-1)*(pca_data.t()*pca_data);
    }
    
    if(old_cov.rows==0)old_cov=new_cov.clone();
    {
      ZoneScopedN("pca");
    // calc PCA
    SVD::compute((1.0-pca_rate)*old_cov+pca_rate*new_cov, w, u, vt);
    }

    Mat proj_vars;
    {
      ZoneScopedN("extract");
    // extract the projection matrix
    proj_matrix=u(Rect(0,0,compressed_sz,src.channels())).clone();
    proj_vars=Mat::eye(compressed_sz,compressed_sz,proj_matrix.type());
    }

    {
      ZoneScopedN("update");
     // #pragma omp parallel for
    for(int i=0;i<compressed_sz;i++){
      proj_vars.at<float>(i,i)=w.at<float>(i);
    }
    }

    {
      ZoneScopedN("covar update");
    // update the covariance matrix
    old_cov=(1.0-pca_rate)*old_cov+pca_rate*proj_matrix*proj_vars*proj_matrix.t();
    }
  }

  /*
   * compress the features
   */
  void inline FastTrackerFFTW::compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const {
    {
    ZoneScopedN("compr_reshape");
    data=src.reshape(1,src.rows*src.cols);
    }
    {
    ZoneScopedN("compr_mult");
    //std::cout << proj_matrix.rows << "," << proj_matrix.cols << "\n";
    //std::cout << src.rows << "," << src.cols << "\n";
    //std::cout << data.rows << "x" << data.cols << " @ " << proj_matrix.rows << "x" << proj_matrix.cols << "\n";
    compressed=data*proj_matrix;
    }
    {
    ZoneScopedN("compr_clone");
    dest=compressed.reshape(proj_matrix.cols,src.rows).clone();
    }
  }

  /*
   * obtain the patch and apply hann window filter to it
   */
  bool FastTrackerFFTW::getSubWindow(const Mat img, const Rect _roi, Mat& feat, Mat& patch, FastTrackerFFTW::MODE desc) const {
    ZoneScopedN("ftmp getSubWnd");
    Rect region=_roi;

    // return false if roi is outside the image
    if ((roi & Rect2d(0, 0, img.cols, img.rows)).empty())
        return false;

    // extract patch inside the image
    if(_roi.x<0){region.x=0;region.width+=_roi.x;}
    if(_roi.y<0){region.y=0;region.height+=_roi.y;}
    if(_roi.x+_roi.width>img.cols)region.width=img.cols-_roi.x;
    if(_roi.y+_roi.height>img.rows)region.height=img.rows-_roi.y;
    if(region.width>img.cols)region.width=img.cols;
    if(region.height>img.rows)region.height=img.rows;

    // return false if region is empty
    if (region.empty())
        return false;

    patch=img(region).clone();

    // add some padding to compensate when the patch is outside image border
    int addTop,addBottom, addLeft, addRight;
    addTop=region.y-_roi.y;
    addBottom=(_roi.height+_roi.y>img.rows?_roi.height+_roi.y-img.rows:0);
    addLeft=region.x-_roi.x;
    addRight=(_roi.width+_roi.x>img.cols?_roi.width+_roi.x-img.cols:0);

    copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);
    if(patch.rows==0 || patch.cols==0)return false;

    // extract the desired descriptors
    switch(desc){
      case CN:
        CV_Assert(img.channels() == 3);
        extractCN(patch,feat);
        feat=feat.mul(hann_cn); // hann window filter
        break;
      default: // GRAY
        if(img.channels()>1)
          cvtColor(patch,feat, COLOR_BGR2GRAY);
        else
          feat=patch;
        //feat.convertTo(feat,CV_32F);
        feat.convertTo(feat,CV_32F, 1.0/255.0, -0.5);
        //feat=feat/255.0-0.5; // normalize to range -0.5 .. 0.5
        feat=feat.mul(hann); // hann window filter
        break;
    }

    return true;

  }

  /*
   * get feature using external function
   */
  bool FastTrackerFFTW::getSubWindow(const Mat img, const Rect _roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const{
    ZoneScopedN("ftmp getSubWnd");

    // return false if roi is outside the image
    if((_roi.x+_roi.width<0)
      ||(_roi.y+_roi.height<0)
      ||(_roi.x>=img.cols)
      ||(_roi.y>=img.rows)
    )return false;

    f(img, _roi, feat);

    if(_roi.width != feat.cols || _roi.height != feat.rows){
      printf("error in customized function of features extractor!\n");
      printf("Rules: roi.width==feat.cols && roi.height = feat.rows \n");
    }

    Mat hann_win;
    std::vector<Mat> _layers;

    #pragma omp parallel for 
    for(int i=0;i<feat.channels();i++)
      _layers.push_back(hann);

    merge(_layers, hann_win);

    feat=feat.mul(hann_win); // hann window filter

    return true;
  }

  /* Convert BGR to ColorNames
   */
  void FastTrackerFFTW::extractCN(Mat patch_data, Mat & cnFeatures) const {
    ZoneScopedN("ftmp extractCn");
    Vec3b & pixel = patch_data.at<Vec3b>(0,0);
    unsigned index;

    if(cnFeatures.type() != CV_32FC(10))
      cnFeatures = Mat::zeros(patch_data.rows,patch_data.cols,CV_32FC(10));

    // perform the loops in parallel using OpenMP
    // and collapsed the loops to avoid overhead

    // for loop variables
    const int batch_size = 2;
    const int patch_area = patch_data.rows*patch_data.cols;
    #pragma omp parallel for
    for(int idx=0;idx<patch_area;idx+=batch_size){
      for(int offset=0; offset<batch_size; offset++){
        if(idx+offset >= patch_area) break;
        int h = idx+offset;
        int i = h/patch_data.cols;
        int j = h%patch_data.cols;
        Vec3b pixel = patch_data.at<Vec3b>(i,j);
        unsigned index=(unsigned)(floor((float)pixel[2]/8)+32*floor((float)pixel[1]/8)+32*32*floor((float)pixel[0]/8));
        // for(int k=0;k<10;k++)
        //   cnFeatures.at<Vec<float,10> >(i,j)[k] = ColorNames[index][k];
        //auto t = cnFeatures.at<Vec<float, 10> >(i, j);
        Vec<float, 10> t;
        t[0] = ColorNames[index][0];
        t[1] = ColorNames[index][1];
        t[2] = ColorNames[index][2];
        t[3] = ColorNames[index][3];
        t[4] = ColorNames[index][4];
        t[5] = ColorNames[index][5];
        t[6] = ColorNames[index][6];
        t[7] = ColorNames[index][7];
        t[8] = ColorNames[index][8];
        t[9] = ColorNames[index][9];
        cnFeatures.at<Vec<float,10> >(i,j) = t;
      }
    }
    // const int batch_size = 4;
    // const int patch_area = patch_data.rows*patch_data.cols;
    // #pragma omp parallel for
    // for(int idx=0;idx<patch_area;idx+=batch_size){
    //   unsigned indices[batch_size];
    //   int is[batch_size];
    //   int js[batch_size];
    //   for(int offset=0; offset<batch_size; offset++){
    //     if(idx+offset >= patch_area) break;
    //     int h = idx+offset;
    //     int i = h/patch_data.cols;
    //     is[offset] = i;
    //     int j = h%patch_data.cols;
    //     js[offset] = j;
    //     Vec3b pixel = patch_data.at<Vec3b>(i,j);
    //     indices[offset] = (unsigned)(floor((float)pixel[2]/8)+32*floor((float)pixel[1]/8)+32*32*floor((float)pixel[0]/8));
    //   }
    //     // for(int k=0;k<10;k++)
    //     //   cnFeatures.at<Vec<float,10> >(i,j)[k] = ColorNames[index][k];
    //     //auto t = cnFeatures.at<Vec<float, 10> >(i, j);
    //   for(int offset=0; offset<batch_size; offset++){
    //     Vec<float, 10> t;
    //     unsigned index = indices[offset];
    //     t[0] = ColorNames[index][0];
    //     t[1] = ColorNames[index][1];
    //     t[2] = ColorNames[index][2];
    //     t[3] = ColorNames[index][3];
    //     t[4] = ColorNames[index][4];
    //     t[5] = ColorNames[index][5];
    //     t[6] = ColorNames[index][6];
    //     t[7] = ColorNames[index][7];
    //     t[8] = ColorNames[index][8];
    //     t[9] = ColorNames[index][9];
    //     cnFeatures.at<Vec<float,10> >(is[offset],js[offset]) = t;
    //   }
    // }
    // for(int idx=0;idx<patch_data.rows*patch_data.cols;idx++){
    //   int i = idx/patch_data.cols;
    //   int j = idx%patch_data.cols;

    //   Vec3b & pixel=patch_data.at<Vec3b>(i,j);
    //   unsigned index=(unsigned)(floor((float)pixel[2]/8)+32*floor((float)pixel[1]/8)+32*32*floor((float)pixel[0]/8));

    //   //copy the values
    //   for(int _k=0;_k<10;_k++){
    //     cnFeatures.at<Vec<float,10> >(i,j)[_k]=ColorNames[index][_k];
    //   }
    // }
  }

  /*
   *  dense gauss kernel function
   */
  void FastTrackerFFTW::denseGaussKernel(const float sigma, const Mat x_data, const Mat y_data, Mat & k_data,
                                        std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) {
    ZoneScopedN("ftmp denseGauss");
    double normX, normY;

    // fft2(x_data,xf_data,layers_data);
    // fft2(y_data,yf_data,layers_data);
    fftw_fft2(x_data,xf_data,layers_data);
    fftw_fft2(y_data,yf_data,layers_data);

    normX=norm(x_data);
    normX*=normX;
    normY=norm(y_data);
    normY*=normY;

    pixelWiseMult(xf_data,yf_data,xyf_v,0,true);
    sumChannels(xyf_v,xyf);
    ifft2(xyf,xyf);

    if(params.wrap_kernel){
      shiftRows(xyf, x_data.rows/2);
      shiftCols(xyf, x_data.cols/2);
    }

    //(xx + yy - 2 * xy) / numel(x)
    //xy=(normX+normY-2*xyf)/(x_data.rows*x_data.cols*x_data.channels());
    const double normXY = normX+normY-2;
    const double normXYDiv = 1.0/(x_data.rows*x_data.cols*x_data.channels());
    parallelElementWiseMult(xyf, xy, normXY*normXYDiv, 10);

    // TODO: check wether we really need thresholding or not
    //threshold(xy,xy,0.0,0.0,THRESH_TOZERO);//max(0, (xx + yy - 2 * xy) / numel(x))
    {
      ZoneScopedN("thresholding");
    #pragma omp parallel for private(xy)
    for(int i=0;i<xy.rows;i++){
      for(int j=0;j<xy.cols;j++){
        if(xy.at<float>(i,j)<0.0)
            xy.at<float>(i,j)=0.0;
      }
    }
    }

    float sig=-1.0f/(sigma*sigma);
    xy=sig*xy;
    exp(xy,k_data);

  }

  /* CIRCULAR SHIFT Function
   * http://stackoverflow.com/questions/10420454/shift-like-matlab-function-rows-or-columns-of-a-matrix-in-opencv
   */
  // circular shift one row from up to down
  void FastTrackerFFTW::shiftRows(Mat& mat) const {
      ZoneScopedN("ftmp shiftRows1");

      // Mat temp;
      // Mat m;
      // int k = (mat.rows-1);
      // mat.row(k).copyTo(temp);

      // #pragma omp parallel for private(m)
      // for(int i = k; i >= 0; i--) {
      //   auto targetIdx = std::abs((i-1) % k);
      //   m = mat.row(i);  // copy the current row into m
      //   mat.row(targetIdx).copyTo(m);  // ...then overwrite it with row i-1?
      // }
      // temp.copyTo(m);
      Mat temp;
      Mat m;
      int _k = (mat.rows-1);
      mat.row(_k).copyTo(temp);
      for(; _k > 0 ; _k-- ) {
        m = mat.row(_k);
        mat.row(_k-1).copyTo(m);
      }
      m = mat.row(0);
      temp.copyTo(m);

  }

  // circular shift n rows from up to down if n > 0, -n rows from down to up if n < 0
  void FastTrackerFFTW::shiftRows(Mat& mat, int n) const {
    ZoneScopedN("ftmp shiftRowsN");
      if( n < 0 ) {
        n = -n;
        flip(mat,mat,0);
        
        #pragma omp parallel for private(mat)
        for(int _k=0; _k < n;_k++) {
          shiftRows(mat);
        }
        flip(mat,mat,0);
      }else{
        
        #pragma omp parallel for private(mat)
        for(int _k=0; _k < n;_k++) {
          shiftRows(mat);
        }
      }
  }

  //circular shift n columns from left to right if n > 0, -n columns from right to left if n < 0
  void FastTrackerFFTW::shiftCols(Mat& mat, int n) const {
    ZoneScopedN("ftmp shiftCols");
      if(n < 0){
        n = -n;
        flip(mat,mat,1);
        transpose(mat,mat);
        shiftRows(mat,n);
        transpose(mat,mat);
        flip(mat,mat,1);
      }else{
        transpose(mat,mat);
        shiftRows(mat,n);
        transpose(mat,mat);
      }
  }

  /*
   * calculate the detection response
   */
  void FastTrackerFFTW::calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const {
    ZoneScopedN("ftmp calcResp");
    //alpha f--> 2channels ; k --> 1 channel;
    mulSpectrums(alphaf_data,kf_data,spec_data,0,false);
    ifft2(spec_data,response_data);
  }

  /*
   * calculate the detection response for splitted form
   */
  void FastTrackerFFTW::calcResponse(const Mat alphaf_data, const Mat _alphaf_den, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const {
    ZoneScopedN("ftmp calcResp");

    mulSpectrums(alphaf_data,kf_data,spec_data,0,false);

    //z=(a+bi)/(c+di)=[(ac+bd)+i(bc-ad)]/(c^2+d^2)
    float den;

    #pragma omp parallel for private(den)
    for(int i=0;i<kf_data.rows;i++){
      for(int j=0;j<kf_data.cols;j++){
        den=1.0f/(_alphaf_den.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[0]+_alphaf_den.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[1]);
        spec2_data.at<Vec2f>(i,j)[0]=
          (spec_data.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[0]+spec_data.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[1])*den;
        spec2_data.at<Vec2f>(i,j)[1]=
          (spec_data.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[0]-spec_data.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[1])*den;
      }
    }

    ifft2(spec2_data,response_data);
  }

  void FastTrackerFFTW::setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func){
    if(pca_func){
      extractor_pca.push_back(f);
      use_custom_extractor_pca = true;
    }else{
      extractor_npca.push_back(f);
      use_custom_extractor_npca = true;
    }
  }
  /*----------------------------------------------------------------------*/

FastTrackerFFTW::Params::Params()
{
  detect_thresh = 0.5f;
  sigma=0.2f;
  lambda=0.0001f;
  interp_factor=0.075f;
  output_sigma_factor=1.0f / 16.0f;
  resize=true;
  max_patch_size=80*80;
  split_coeff=true;
  wrap_kernel=false;
  desc_npca = GRAY;
  desc_pca = CN;

  //feature compression
  compress_feature=true;
  compressed_size=2;
  pca_learning_rate=0.15f;
}

} // namespace
