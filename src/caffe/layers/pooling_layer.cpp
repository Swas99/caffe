#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/pool.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

template <>
void PoolingLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <>
void PoolingLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  float* top_mask = NULL;
  int num = bottom[0]->num();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
  {
    int* mask = NULL;  // suppress warnings about uninitalized variables
    if (!use_top_mask) mask = max_idx_.mutable_cpu_data();
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();

      // The main loop
#pragma omp parallel for collapse(2)
      for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels_; ++c) {
          // compute offset
          const float *bottom_data_cur = bottom_data + bottom[0]->offset(0, 1)*(channels_*n + c);
          float *top_data_cur = top_data + top[0]->offset(0, 1)*(channels_*n + c);
          int *mask_cur = mask + top[0]->offset(0, 1)*(channels_*n + c);
          float *top_mask_cur = top_mask + top[0]->offset(0, 1)*(channels_*n + c);

          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = min(hstart + kernel_h_, height_);
              int wend = min(wstart + kernel_w_, width_);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              float maximum = -FLT_MAX;
              int mask = -1;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width_ + w;
                  if (bottom_data_cur[index] > maximum) {
                    maximum = bottom_data_cur[index];
                    mask = static_cast<float>(index);
                  }
                }
              }
              const int pool_index = ph * pooled_width_ + pw;
              top_data_cur[pool_index] = maximum;
              top_data_cur[pool_index] = mask;
            }
          }
        } // for each channel
      } // for each input layer
    }
    else { // !use_top_mask
      // JSP: typical path, stride=2 kernel=3

      // The main loop
#pragma omp parallel for
      for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels_; ++c) {
          // compute offset
          const float *bottom_data_cur = bottom_data + bottom[0]->offset(0, 1)*(channels_*n + c);
          float *top_data_cur = top_data + top[0]->offset(0, 1)*(channels_*n + c);
          int *mask_cur = mask + top[0]->offset(0, 1)*(channels_*n + c);

          if (stride_h_ == stride_w_ && kernel_h_ == kernel_w_ && pad_h_ == pad_w_ && height_ == width_) {
            if (3 == kernel_w_) {
              if (2 == stride_h_ && 0 == pad_h_) {
                if (112 == height_) {
                  pool_<2, 2, 3, 3, 0, 0, 112, 112>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
                else if (56 == height_) {
                  pool_<2, 2, 3, 3, 0, 0, 56, 56>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
                else if (28 == height_) {
                  pool_<2, 2, 3, 3, 0, 0, 28, 28>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
                else if (14 == height_) {
                  pool_<2, 2, 3, 3, 0, 0, 14, 14>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
                // AlexNet
                else if (55 == height_) {
                  pool_<2, 2, 3, 3, 0, 0, 55, 55>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
              }
              else if (1 == stride_h_ && 1 == pad_h_) {
                if (28 == height_) {
                  pool_<1, 1, 3, 3, 1, 1, 28, 28>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
                else if (14 == height_) {
                  pool_<1, 1, 3, 3, 1, 1, 14, 14>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
                else if (7 == height_) {
                  pool_<1, 1, 3, 3, 1, 1, 7, 7>(bottom_data_cur, top_data_cur, mask_cur);
                  continue;
                }
              }
            }
          }

          for (int ph = 0; ph < pooled_height_; ++ph) {
            int hstart = ph * stride_h_ - pad_h_;
            int hend = min(hstart + kernel_h_, height_);
            hstart = max(hstart, 0);

            for (int pw = 0; pw < pooled_width_; ++pw) {
              int wstart = pw * stride_w_ - pad_w_;
              int wend = min(wstart + kernel_w_, width_);
              wstart = max(wstart, 0);
              float maximum = -FLT_MAX;
              int mask = -1;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width_ + w;
                  if (bottom_data_cur[index] > maximum) {
                    maximum = bottom_data_cur[index];
                    mask = index;
                  }
                }
              }
              const int pool_index = ph * pooled_width_ + pw;
              top_data_cur[pool_index] = maximum;
              mask_cur[pool_index] = mask;
            }
          }
        } // for each channel
      } // for each input layer
    }
    break;
  }
  case PoolingParameter_PoolMethod_AVE:
#pragma omp parallel for
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
#pragma omp parallel for collapse(2)
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels_; ++c) {
        const float *bottom_data_cur = bottom_data + bottom[0]->offset(0, 1)*(channels_*n + c);
        float *top_data_cur = top_data + top[0]->offset(0, 1)*(channels_*n + c);

        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data_cur[ph * pooled_width_ + pw] +=
                    bottom_data_cur[h * width_ + w];
              }
            }
            top_data_cur[ph * pooled_width_ + pw] /= pool_size;
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
