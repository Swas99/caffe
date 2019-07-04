#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
BaseConvolutionLayer<Dtype>::BaseConvolutionLayer(const LayerParameter& param)
    : Layer<Dtype>(param), input_padded_(NULL), output_scratch_(NULL)
{
  //is_sparse_format_weights_ = false;
  is_concatenating_weights_features_ = false;
}

template <typename Dtype>
BaseConvolutionLayer<Dtype>::~BaseConvolutionLayer()
{
  free(input_padded_);
  free(output_scratch_);

  for (int i = 0; i < weight_rowptr_.size(); ++i) {
    free(weight_rowptr_[i]);
    free(weight_colidx_[i]);
    free(weight_values_[i]);
  }

  for (int i = 0; i < weight_rowptr_blocked_.size(); ++i) {
    free(weight_rowptr_blocked_[i]);
    free(weight_colidx_blocked_[i]);
    free(weight_values_blocked_[i]);
  }
}

bool barrier_initialized = false;

template <>
void BaseConvolutionLayer<double>::WeightAlign(){
  NOT_IMPLEMENTED;
}

template <>
void BaseConvolutionLayer<float>::WeightAlign()
{
  cpu::OpenMpManager::getThreadGroupBarriers(this->num_);

  CHECK_EQ(this->blobs_[0]->num_axes(),4);//caffe now supports any dimension
  //is_sparse_format_weights_ = false;
  const LayerParameter& layerparam = this->layer_param();
  LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity(Solver<float>::getMeasureThreshold());

  ConvolutionParameter conv_param = layerparam.convolution_param();
  if (conv_param.dump_parameter()) {
    this->blobs_[0]->WriteToNistMMIOSparse(layerparam.name()+".mtx");
  }

  const int M = this->blobs_[0]->shape(0)/group_;
  const int N = this->blobs_[0]->count(1,4);
  const int weight_offset = this->blobs_[0]->count()/group_;
  const int row_offset = this->blobs_[0]->shape(0)/group_ + 1;
  int masked_col_num = 0;
  int left_cols = 0;
  float group_sparsity = 0;

  int height = conv_input_shape_.cpu_data()[1];
  int width = conv_input_shape_.cpu_data()[2];
  int pad_h = pad_.cpu_data()[0];
  int pad_w = pad_.cpu_data()[1];

  if (caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV == conv_param.conv_mode()) {
    int input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w) + VLEN - 1;
    posix_memalign((void **)&input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);
    memset(input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);
  }
  else if (caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV == conv_param.conv_mode()) {
    int input_padded_len = conv_in_channels_ * (height + 2*pad_h) * (width + 2*pad_w);
    posix_memalign((void **)&input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);
    memset(input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);
  }

  switch(conv_param.conv_mode()){
    case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM:
      LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CCNMM";
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_if_all_zero(M,
            N,
            this->blobs_[0]->cpu_data() + weight_offset * g,
            col_buf_mask_.mutable_cpu_data() + N * g);
      }
      masked_col_num = 0;
      for(int idx=0; idx<col_buf_mask_.count();++idx){
        if(col_buf_mask_.cpu_data()[idx]){
          masked_col_num++;
        }
      }
      group_sparsity = (float)masked_col_num/(float)col_buf_mask_.count();
      LOG(INFO) << Layer<float>::layer_param().name() << " column sparsity: " << group_sparsity;
      is_concatenating_weights_features_ = true;

      // compress weight matrix
      left_cols = 0;
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_del_zero_cols(conv_out_channels_ /group_,
            kernel_dim_ ,
            this->blobs_[0]->cpu_data() + weight_offset_ * g,
            squeezed_weight_buffer_.mutable_cpu_data() + weight_offset_ * g,
            &left_cols,
            col_buf_mask_.cpu_data() + kernel_dim_ * g );
        left_columns_.push_back(left_cols);
      }
      break;
    case caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV:
      {
        LOG(INFO)<<"ConvolutionParameter_ConvMode_DIRECT_SCONV";

        int kernel_h = kernel_shape_.cpu_data()[0];
        int kernel_w = kernel_shape_.cpu_data()[1];

        int temp_nnz = 0;
        for (int g = 0; g < group_; ++g) {
          for (int i = 0; i < M*N; ++i) {
            if (this->blobs_[0]->cpu_data()[weight_offset*g + i] != 0) ++temp_nnz;
          }
        }
        int col_block_size = get_col_major_ic_block(temp_nnz/group_, M, conv_in_channels_/group_);
        assert(conv_in_channels_/group_%col_block_size == 0);

        int ncolblocks = conv_in_channels_/col_block_size;
        assert(ncolblocks >= 1);
        LOG(INFO) << "ncolblocks " << ncolblocks;
        weight_rowptr_blocked_.resize(ncolblocks);
        weight_colidx_blocked_.resize(ncolblocks);
        weight_values_blocked_.resize(ncolblocks);
        std::vector<int> nnzs_of_col_blocks(ncolblocks, 0);

        weight_rowptr_.resize(group_);
        weight_colidx_.resize(group_);
        weight_values_.resize(group_);

        for (int g = 0; g < group_; ++g) {
          int nnz = 0;
          for (int i = 0; i < M*N; ++i) {
            if (this->blobs_[0]->cpu_data()[weight_offset*g + i] != 0) ++nnz;
          }

          posix_memalign((void **)&weight_rowptr_[g], 4096, sizeof(int)*(M + 1));
          posix_memalign((void **)&weight_colidx_[g], 4096, sizeof(int)*nnz);
          posix_memalign((void **)&weight_values_[g], 4096, sizeof(float)*nnz);

          // first create a CSR matrix as for LOWERED_CSRMM
          caffe_cpu_sparse_dense2csr(M, N,
              this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
              weight_values_[g],
              weight_colidx_[g],
              weight_rowptr_[g]);

          // declare variables for sparsity statistics
          vector<vector<int> > nnz_per_channel_pair(M);
          for(int i = 0; i < M; ++i) {
            nnz_per_channel_pair[i] = vector<int>(conv_in_channels_, 0);
          }
          vector<int> nnz_per_oc_fiber(N, 0);
          assert(N == conv_in_channels_/group_*kernel_h*kernel_w);
          int num_of_non_zero_kernels = 0;
          int num_of_non_zero_out_channels = 0;

          const int *rowptr = weight_rowptr_[g];
          assert(nnz == rowptr[M]);

          int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);
          assert(conv_in_channels_/group_%col_major_ic_block == 0);
          LOG(INFO) << "col_major_ic_block = " << col_major_ic_block;

          // transform the indices for direct convolution
          int *colidx = weight_colidx_[g];
          for (int oc = 0; oc < M; ++oc) {
            for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
              int col = colidx[j];

              int kernel_col = col%kernel_w;
              int kernel_row = (col/kernel_w)%kernel_h;
              int ic = col/(kernel_w*kernel_h);
              assert(ic < conv_in_channels_/group_);

              colidx[j] = (ic*(height + pad_h) + kernel_row)*(width + pad_w) + kernel_col;

              int bcol = ic/col_block_size + ncolblocks/group_*g;
              ++nnzs_of_col_blocks[bcol];

              ++nnz_per_channel_pair[oc][ic];
              ++nnz_per_oc_fiber[col];
            }
            if (rowptr[oc + 1] > rowptr[oc]) {
              num_of_non_zero_out_channels++;
            }

            for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
              if (nnz_per_channel_pair[oc][in_channel] != 0) {
                ++num_of_non_zero_kernels;
              }
            }
          }

          int num_of_non_zero_oc_fibers = 0;
          for (int i = 0 ; i < N; ++i) {
            if (nnz_per_oc_fiber[i] > 0) ++num_of_non_zero_oc_fibers;
          }

          std::vector<int> kernel_non_zero_hist(kernel_w*kernel_h, 0);
          for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
            int cnt = 0;
            for (int i = in_channel*kernel_w*kernel_h; i < (in_channel + 1)*kernel_w*kernel_h; ++i) {
              kernel_non_zero_hist[i - in_channel*kernel_w*kernel_h] += nnz_per_oc_fiber[i];
            }
          }

          std::stringstream stream;
          stream << "kernel_non_zero_hist = ";
          for (int i = 0; i < kernel_w*kernel_h; ++i) {
            stream << i << ":" << kernel_non_zero_hist[i] << " ";
          }
          LOG(INFO) << stream.str();

          LOG(INFO) << "oc-mode fiber sparsity " << 1 - (double)num_of_non_zero_oc_fibers/N;
          LOG(INFO) << "oc-mode slice sparsity " << 1 - (double)num_of_non_zero_out_channels/M;
          LOG(INFO) << "k-mode fiber sparsity " << 1 - (double)num_of_non_zero_kernels/(M*(conv_in_channels_/group_));
          LOG(INFO) << "nnz = " << nnz;
        } // for each group

        for (int i = 0; i < ncolblocks; ++i) {
          posix_memalign((void **)&weight_rowptr_blocked_[i], 4096, sizeof(int)*(M + 1));
          posix_memalign((void **)&weight_colidx_blocked_[i], 4096, sizeof(int)*nnzs_of_col_blocks[i]);
          posix_memalign((void **)&weight_values_blocked_[i], 4096, sizeof(float)*nnzs_of_col_blocks[i]);
          nnzs_of_col_blocks[i] = 0;
          weight_rowptr_blocked_[i][0] = 0;
        }

        int stride_h = stride_.cpu_data()[0];
        int stride_w = stride_.cpu_data()[1];
        int dilation_h = dilation_.cpu_data()[0];
        int dilation_w = dilation_.cpu_data()[1];

        const int output_h = (height + 2 * pad_h -
            (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w -
            (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        const int SCRATCH_SIZE_PER_IC = output_h*((output_w + 16 - 1)/16*16);

        int max_col_major_ic_block = 0;
        for (int g = 0; g < group_; ++g) {
          const int *rowptr = weight_rowptr_[g];
          int *colidx = weight_colidx_[g];
          float *values = weight_values_[g];

          int nnz = rowptr[M];
          int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);
          max_col_major_ic_block = std::max(max_col_major_ic_block, col_major_ic_block);

          for (int oc = 0; oc < M; ++oc) {
            for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
              int c = colidx[j];

              int kernel_col = c%(width + pad_w);
              int kernel_row = c/(width + pad_w)%(height + pad_h);
              int ic = c/(width + pad_w)/(height + pad_h);
              int bcol = ic/col_block_size + ncolblocks/group_*g;

              weight_colidx_blocked_[bcol][nnzs_of_col_blocks[bcol]] = c;
              weight_values_blocked_[bcol][nnzs_of_col_blocks[bcol]] = values[j];
              nnzs_of_col_blocks[bcol]++;
            }

            for (int i = ncolblocks/group_*g; i < ncolblocks/group_*(g + 1); ++i) {
              weight_rowptr_blocked_[i][oc + 1] = nnzs_of_col_blocks[i];
            }
          }
        } // for each group

        posix_memalign((void **)&output_scratch_, 4096, sizeof(float)*OC_BLOCK*output_h*((output_w + 16 - 1)/16*16)*omp_get_max_threads());

        break;
      }
      LOG(INFO)<<"ConvolutionParameter ConvMode: DEFAULT";
      break;
  }

  //disconnect connections
  if( layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_ELTWISE ){
    LOG(INFO)<<"all zero weights of "<<layerparam.name()<<" are frozen";
    this->blobs_[0]->Disconnect(Blob<float>::ELTWISE, Solver<float>::getPruneThreshold());
  }else if(layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_GRPWISE){
    LOG(INFO)<<"weights lying in all-zero groups of "<<layerparam.name()<<" are frozen";
    this->blobs_[0]->Disconnect(Blob<float>::GRPWISE, Solver<float>::getPruneThreshold(), group_);
  }

  if (conv_param.parameter_sparsity_pattern_histogram() && std::string(this->type()) != "Winograd") {
    int kernel_h = kernel_shape_.cpu_data()[0];
    int kernel_w = kernel_shape_.cpu_data()[1];

    CHECK(weight_offset == M*conv_in_channels_/group_*kernel_h*kernel_w);
    CHECK(N == conv_in_channels_/group_*kernel_h*kernel_w);

    map<unsigned long long, int> hist;

    for (int g = 0; g < group_; ++g) {
      for (int oc = 0; oc < M; ++oc) {
        for (int ic = 0; ic < conv_in_channels_/group_; ++ic) {
          int pattern = 0;
          for (int k = 0; k < kernel_h*kernel_w; ++k) {
            if (this->blobs_[0]->cpu_data()[weight_offset*g + (oc*conv_in_channels_/group_ + ic)*kernel_h*kernel_w + k] != 0) {
              pattern |= (1 << k);
            }
          }
          if (hist.find(pattern) == hist.end()) {
            hist[pattern] = 0;
          }
          ++hist[pattern];
        } // for each input channel
      } // for each output channel
    } // for each group

    set<pair<int, unsigned long long> > inverseHist;
    for (map<unsigned long long, int>::iterator pattern = hist.begin(); pattern != hist.end(); ++pattern) {
      inverseHist.insert(make_pair<int, unsigned long long>(pattern->second, pattern->first));
    }

    fprintf(stderr, "total = %d\n", M*conv_in_channels_);
    for (set<pair<int, unsigned long long> >::reverse_iterator pattern = inverseHist.rbegin(); pattern != inverseHist.rend(); ++pattern) {
      fprintf(stderr, "%d\n", pattern->first);
      for (int h = 0; h < kernel_h; ++h) {
        for (int w = 0; w < kernel_w; ++w) {
          fprintf(stderr, "%d ", (pattern->second & (1 << (h*kernel_w + w))) != 0);
        }
        fprintf(stderr, "\n");
      }
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
        << " vs. bottom[" << bottom_id << "]: "
        << bottom[bottom_id]->shape_string();
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
