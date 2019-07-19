#include <vector>

#include "caffe/layers/winograd_layer.hpp"
#include "caffe/util/winograd.hpp"

namespace caffe {

template <typename Dtype>
__global__ void winograd_input_im2col_gpu_kernel(
  const int n,
  const Dtype *data, Dtype *col_buff,
  int height, int width,
  int pad_h, int pad_w,
  int ntiles_h, int ntiles_w,
  int tile_h_in, int tile_w_in,
  int tile_h_out, int tile_w_out,
  int nchannels, int batch_size)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_in;
    const int y = index/tile_w_in%tile_h_in;
    const int tile_w = index/tile_w_in/tile_h_in%ntiles_w;
    const int tile_h = index/tile_w_in/tile_h_in/ntiles_w%ntiles_h;
    const int c = index/tile_w_in/tile_h_in/ntiles_w/ntiles_h%nchannels;
    const int image_idx = index/tile_w_in/tile_h_in/ntiles_w/ntiles_h/nchannels;

    int in_y = tile_h*tile_h_out + y - pad_h;
    int in_x = tile_w*tile_w_out + x - pad_w;

    if (in_y < 0 || in_x < 0 || in_y >= height || in_x >= width) {
      col_buff[((((image_idx + c*batch_size)*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x] = 0;
    }
    else {
      col_buff[((((image_idx + c*batch_size)*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x] = data[((image_idx*nchannels + c)*height + in_y)*width + in_x];
    }
  }
}

template <typename Dtype>
__global__ void winograd_output_col2im_gpu_kernel(
  const int n,
  const Dtype *col_buff, Dtype *data,
  int output_h, int output_w,
  int ntiles_h, int ntiles_w,
  int tile_h_out, int tile_w_out,
  int nchannels, int batch_size)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_out;
    const int y = index/tile_w_out%tile_h_out;
    const int tile_w = index/tile_w_out/tile_h_out%ntiles_w;
    const int tile_h = index/tile_w_out/tile_h_out/ntiles_w%ntiles_h;
    const int c = index/tile_w_out/tile_h_out/ntiles_w/ntiles_h%nchannels;
    const int image_idx = index/tile_w_out/tile_h_out/ntiles_w/ntiles_h/nchannels;

    int out_y = tile_h*tile_h_out + y;
    int out_x = tile_w*tile_w_out + x;

    if (out_y < output_h && out_x < output_w) {
      data[((image_idx*nchannels + c)*output_h + out_y)*output_w + out_x] =
          col_buff[((((image_idx + c*batch_size)*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x];
    }
  }
}

template <typename Dtype>
__global__ void winograd_output_im2col_gpu_kernel(
  const int n,
  const Dtype *data, Dtype *col_buff,
  int output_h, int output_w,
  int ntiles_h, int ntiles_w,
  int tile_h_out, int tile_w_out,
  int nchannels, int batch_size)
{
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_out;
    const int y = index/tile_w_out%tile_h_out;
    const int tile_w = index/tile_w_out/tile_h_out%ntiles_w;
    const int tile_h = index/tile_w_out/tile_h_out/ntiles_w%ntiles_h;
    const int c = index/tile_w_out/tile_h_out/ntiles_w/ntiles_h%nchannels;
    const int image_idx = index/tile_w_out/tile_h_out/ntiles_w/ntiles_h/nchannels;

    int out_y = tile_h*tile_h_out + y;
    int out_x = tile_w*tile_w_out + x;

    if (out_y < 0 || out_x < 0 || out_y >= output_h || out_x >= output_w) {
      col_buff[((((image_idx + c*batch_size)*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x] = 0;
    }
    else {
      col_buff[((((image_idx + c*batch_size)*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_out + y)*tile_w_out + x] =
          data[((image_idx*nchannels + c)*output_h + out_y)*output_w + out_x];
    }
  }
}

template <typename Dtype>
__global__ void winograd_input_col2im_gpu_kernel(
  const int n,
  const Dtype *col_buff, Dtype *data,
  int height, int width,
  int pad_h, int pad_w,
  int ntiles_h, int ntiles_w,
  int tile_h_in, int tile_w_in,
  int tile_h_out, int tile_w_out,
  int nchannels, int batch_size)
{
  int m = batch_size*nchannels*height*width;
  CUDA_KERNEL_LOOP(index, m) {
  	data[index] = 0;
  }
   
  CUDA_KERNEL_LOOP(index, n) {
    const int x = index%tile_w_in;
    const int y = index/tile_w_in%tile_h_in;
    const int tile_w = index/tile_w_in/tile_h_in%ntiles_w;
    const int tile_h = index/tile_w_in/tile_h_in/ntiles_w%ntiles_h;
    const int c = index/tile_w_in/tile_h_in/ntiles_w/ntiles_h%nchannels;
    const int image_idx = index/tile_w_in/tile_h_in/ntiles_w/ntiles_h/nchannels;

    int in_y = tile_h*tile_h_out + y - pad_h;
    int in_x = tile_w*tile_w_out + x - pad_w;

    if (in_y >= 0 && in_x >= 0 && in_y < height && in_x < width) {
      data[((image_idx*nchannels + c)*height + in_y)*width + in_x] +=
          col_buff[((((image_idx + c*batch_size)*ntiles_h + tile_h)*ntiles_w + tile_w)*tile_h_in + y)*tile_w_in + x];
    }
  }
}

template <>
void WinogradLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

//#define PROFILE_WINOGRAD

template <>
void WinogradLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<float> *AKronA = WinogradAKronA<float>::getInstance(kernel_h);
  WinogradBKronB<float> *BKronB = WinogradBKronB<float>::getInstance(kernel_h);
  WinogradGKronG<float> *GKronG = WinogradGKronG<float>::getInstance(kernel_h);

  const float* weight = this->blobs_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const float* bottom_data = bottom[i]->gpu_data();
    float* top_data = top[i]->mutable_gpu_data();
    
    int M = this->conv_in_channels_*ntiles_h_*ntiles_w_;
    int num_kernels = this->conv_in_channels_*this->num_*ntiles_h_*ntiles_w_*tile_h_in_*tile_w_in_;
    int height = this->conv_input_shape_.cpu_data()[1], width = this->conv_input_shape_.cpu_data()[2];
    int pad_h = this->pad_.cpu_data()[0], pad_w = this->pad_.cpu_data()[1];

    winograd_input_im2col_gpu_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels),
                                              CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, bottom_data, temp2_.mutable_gpu_data(),
      height, width,
      pad_h, pad_w,
      ntiles_h_, ntiles_w_,
      tile_h_in_, tile_w_in_,
      tile_h_out_, tile_w_out_,
      this->conv_in_channels_, this->num_);
    CUDA_POST_KERNEL_CHECK;


      //here
  }
}

template <>
void WinogradLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <>
void WinogradLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {

  int kernel_h = this->kernel_shape_.cpu_data()[0], kernel_w = this->kernel_shape_.cpu_data()[1];

  WinogradAKronA<float> *AKronA = WinogradAKronA<float>::getInstance(kernel_h);
  WinogradBKronB<float> *BKronB = WinogradBKronB<float>::getInstance(kernel_h);
  WinogradGKronG<float> *GKronG = WinogradGKronG<float>::getInstance(kernel_h);

  const float* weight = this->blobs_[0]->gpu_data();
  float* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  /*const float *weight_cpu = this->blobs_[0]->cpu_data();
  fprintf(stderr, "weight_winograd\n");
  for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
    for (int n = 0; n < this->conv_out_channels_; ++n) {
      for (int c = 0; c < this->conv_in_channels_; ++c) {
        fprintf(stderr, "%g ", weight_cpu[(j*this->conv_out_channels_ + n)*this->conv_in_channels_ + c]);
      }
    }
    fprintf(stderr, "\n");
  }*/

  for (int i = 0; i < top.size(); ++i) {
  //here
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WinogradLayer);

}  // namespace caffe
