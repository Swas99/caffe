#include <algorithm>
#include <vector>
#include <stdio.h>

#include "caffe/filler.hpp"
#include "caffe/layers/base_wino_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/shape_inference.h"


namespace caffe {

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
        // Configure the kernel size, padding, stride, and inputs.
        ConvolutionParameter conv_param = this->layer_param_.convolution_param();
        force_nd_im2col_ = conv_param.force_nd_im2col();
        channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
        const int first_spatial_axis = channel_axis_ + 1;
        const int num_axes = bottom[0]->num_axes();
        num_spatial_axes_ = num_axes - first_spatial_axis;
        CHECK_GE(num_spatial_axes_, 0);
        vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
        vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
        // Setup filter kernel dimensions (kernel_shape_).
        kernel_shape_.Reshape(spatial_dim_blob_shape);
        int *kernel_shape_data = kernel_shape_.mutable_cpu_data();
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
        int *stride_data = stride_.mutable_cpu_data();
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
        int *pad_data = pad_.mutable_cpu_data();
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
        int *dilation_data = dilation_.mutable_cpu_data();
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
        // Special case: im2col is the identity for 1 convolution with stride 1
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

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {
        const int first_spatial_axis = channel_axis_ + 1;
        CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
            << "bottom num_axes may not change.";
        num_ = bottom[0]->count(0, channel_axis_);
        CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
            << "Input size incompatible with convolution kernel.";
        // TODO: generalize to handle inputs of different shapes.
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
            CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
            << "All inputs must have the same shape.";
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
        int *conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
        for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
            if (reverse_dimensions()) {
                conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
            } else {
                conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
            }
        }
        // The im2col result buffer will only hold one image at a time to avoid
        // overly large memory usage. In the special case of 1 convolution
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

    template<typename Dtype>

    void BaseWinogradLayer<Dtype>::winograd_4_4_3_3(Dtype g[3][3], Dtype d[6][6], Dtype Y[4][4]) {
        memset(Y,0, sizeof(Dtype)*16);
        Dtype BTd[6][6];
        for (int i = 0; i < 6; ++i) {
            BTd[0][i] = (d[0][i] * 4) - d[2][i] * 5 + d[4][i];
            BTd[1][i] = -(d[1][i] * 4) - d[2][i] * 4 + d[3][i] + d[4][i];
            BTd[2][i] = (d[1][i] * 4) - d[2][i] * 4 - d[3][i] + d[4][i];
            BTd[3][i] = -(d[1][i] * 2) - d[2][i] + d[3][i] * 2 + d[4][i];
            BTd[4][i] = (d[1][i] * 2) - d[2][i] - d[3][i] * 2 + d[4][i];
            BTd[5][i] = (d[1][i] * 4) - d[3][i] * 5 + d[5][i];
        }
        Dtype V[6][6];
        for (int i = 0; i < 6; ++i) {
            V[i][0] =  (BTd[i][0] * 4) - (BTd[i][2] * 4) - BTd[i][2]     + BTd[i][4];
            V[i][1] = -(BTd[i][1] * 4) - (BTd[i][2] * 4) + BTd[i][3]     + BTd[i][4];
            V[i][2] =  (BTd[i][1] * 4) - (BTd[i][2] * 4) - BTd[i][3]     + BTd[i][4];
            V[i][3] = -(BTd[i][1] * 2) -  BTd[i][2]      + BTd[i][3] * 2 + BTd[i][4];
            V[i][4] =  (BTd[i][1] * 2) -  BTd[i][2]      - BTd[i][3] * 2 + BTd[i][4];
            V[i][5] =  (BTd[i][1] * 4) - (BTd[i][3] * 4) - BTd[i][3]     + BTd[i][5];

        }
        Dtype Gg[6][3];

        for (int i = 0; i < 3; ++i) {
            Gg[0][i] = g[0][i] / 4.;
            Gg[1][i] = ((-g[0][i] - g[1][i] - g[2][i]) / 2) / 3.;
            Gg[2][i] = ((-g[0][i] + g[1][i] - g[2][i]) / 2) / 3.;
            Gg[3][i] = (g[0][i] / 8. + g[1][i] / 4. + g[2][i] / 2) / 3.;
            Gg[4][i] = (g[0][i] / 8. - g[1][i] / 4. + g[2][i] / 2) / 3.;
            Gg[5][i] = g[2][i];
        }
        Dtype U[6][6];
        for (int i = 0; i < 6; ++i) {
            U[i][0] = Gg[i][0] / 4.;
            U[i][1] = ((-Gg[i][0] - Gg[i][1] - Gg[i][2]) / 2) / 3.;
            U[i][2] = ((-Gg[i][0] + Gg[i][1] - Gg[i][2]) / 2) / 3.;
            U[i][3] = (Gg[i][0] / 8. + Gg[i][1] / 4. + Gg[i][2] / 2.) / 3.;
            U[i][4] = (Gg[i][0] / 8. - Gg[i][1] / 4. + Gg[i][2] / 2.) / 3.;
            U[i][5] = Gg[i][2];
        }
        Dtype M[6][6];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                M[i][j] = (U[i][j]) * (V[i][j]);
            }
        }

        // calculate matrix A'M;
        Dtype ATM[4][6];
        for (int i = 0; i < 6; ++i) {
            ATM[0][i] = M[0][i] + M[1][i] +  M[2][i]       + M[3][i]      + M[4][i];
            ATM[1][i] = M[1][i] - M[2][i] + (M[3][i] * 2) - (M[4][i] * 2);
            ATM[2][i] = M[1][i] + M[2][i] + (M[3][i] * 4) + (M[4][i] * 4);
            ATM[3][i] = M[1][i] - M[2][i] + (M[3][i] * 8) - (M[4][i] * 8) + M[5][i];

        }
        for (int i = 0; i < 4; ++i) {
            Y[i][0] = Dtype(ATM[i][0] + ATM[i][1] +  ATM[i][2]      +  ATM[i][3]      + ATM[i][4]);
            Y[i][1] = Dtype(ATM[i][1] - ATM[i][2] + (ATM[i][3] * 2) - (ATM[i][4] * 2));
            Y[i][2] = Dtype(ATM[i][1] + ATM[i][2] + (ATM[i][3] * 4) + (ATM[i][4] * 4));
            Y[i][3] = Dtype(ATM[i][1] - ATM[i][2] + (ATM[i][3] * 8) - (ATM[i][4] * 8) + ATM[i][5]);
        }


    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::flatten(const Dtype out_tile[4][4], Dtype *output, const int tile_ind_x, const int tile_ind_y, const int out_channel,
                 const int out_w, const int out_h) {
        //flatten tile, putting into output and channel-wise sum

        int offset = out_channel * out_h * out_w + tile_ind_y * 4*out_w + tile_ind_x*4;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++)
            {
                *(output+offset+j*out_w+i) += out_tile[i][j];
            }
        }
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::forward_cpu_winograd(const Dtype *input, const Dtype *weights, Dtype *output) {

        //kernel_dim_;
        int in_channels  = conv_in_channels_;
        int out_channels = conv_out_channels_;
        int input_h      = conv_input_shape_.cpu_data()[1];
        int input_w      = conv_input_shape_.cpu_data()[2];
        int kernel_h     = kernel_shape_.cpu_data()[0];
        int kernel_w     = kernel_shape_.cpu_data()[1];
        int pad_h        = pad_.cpu_data()[0];
        int pad_w        = pad_.cpu_data()[1];
        int stride_h     = stride_.cpu_data()[0];
        int stride_w     = stride_.cpu_data()[1];
        int dilation_h   = dilation_.cpu_data()[0];
        int dilation_w   = dilation_.cpu_data()[1];
        int kernel_size  = kernel_h * kernel_w;

        // printf("input_h: %d \n", input_h);
        // printf("input_w: %d \n", input_w);
        // printf("pad_h: %d \n", pad_h);
        // printf("pad_w: %d \n", pad_w);

        if (kernel_h != 3 || kernel_w != 3) {
            LOG(FATAL) << "kernel size must be 3";
        }
        if (pad_h>4||pad_w>4)
        {
            LOG(FATAL) << "padding must less than 4";
        }

        if (group_ > 1) {
            LOG(FATAL) << "multi Groups not implemented ";
        }

        Dtype weight[3][3];  //kernel weight
        Dtype in[6][6]; //input tile
        Dtype out_tile[4][4]; //out put tile
        const int output_h = (input_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (input_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = input_h * input_w;
        const int out_channel_size = output_h*output_w;
        memset(output,0, sizeof(Dtype)*output_h*output_w*out_channels);

        // parameters of padding and tiling
        int tile_num_w = (input_w + 2 * pad_w-6) / 4 + ((input_w + 2 * pad_w-6) % 4 > 0 ? 1 : 0)+1;
        int tile_num_h = (input_h + 2 * pad_h-6) / 4 + ((input_h + 2 * pad_h-6) % 4 > 0 ? 1 : 0)+1;

        int padded_in_w  = 4*tile_num_w+2;
        int padded_in_h  = 4*tile_num_h+2;
        int padded_out_w = 4*tile_num_w;
        int padded_out_h = 4*tile_num_h;
        int padded_channel_size     = padded_in_h*padded_in_w;
        int padded_out_channel_size = padded_out_w*padded_out_h;

        Dtype*padded_out = (Dtype*)malloc(padded_out_channel_size*out_channels* sizeof(Dtype));
        memset(padded_out,0, sizeof(Dtype)*padded_out_channel_size*out_channels);

        //pad 0
        Dtype* padded_input = (Dtype*)malloc(in_channels*padded_channel_size* sizeof(Dtype));
        memset(padded_input,0, sizeof(Dtype)*in_channels*padded_channel_size);

        //copy input to padded_input
        for (int c=0;c<in_channels;c++)
            for (int h=0;h<input_h;h++)
                for (int w=0;w<input_w;w++)
                {
                    *(padded_input+c*padded_channel_size+padded_in_w*(h+pad_h)+w+pad_w) = *(input+c*channel_size+h*input_w+w);
                }

        int tile_x = 0; //tile index x
        int tile_y = 0; //tile index y

        for (int out_channel = 0; out_channel < out_channels; out_channel++) {
            for (int tile_ind_x = 0; tile_ind_x < tile_num_w ; tile_ind_x++)
            {
                for (int tile_ind_y = 0; tile_ind_y < tile_num_h ; tile_ind_y++) {
                    for (int in_channel = 0; in_channel < in_channels; in_channel++) {
                        for (int i = 0; i < kernel_w; i++) 
                        {
                            for (int j = 0; j < kernel_h; j++)
                            {
                                weight[i][j] = *(weights + out_channel * in_channels * kernel_size + in_channel * 3*3 + j * 3 + i);
                            }
                        }

                        tile_x = tile_ind_x * 4;
                        tile_y = tile_ind_y * 4;
                        //insert input tile data
                        for (int i = 0; i < 6; i++) 
                        {
                            for (int j = 0; j < 6; j++)
                            {
                                in[i][j] = *(padded_input + in_channel * padded_in_h*padded_in_w + (tile_y+j)*padded_in_w  + tile_x + i);
                            }
                        }
                        
                        this->winograd_4_4_3_3(weight, in, out_tile);
                        this->flatten(out_tile,padded_out,tile_ind_x,tile_ind_y,out_channel,padded_out_w,padded_out_h);
                    }
                }
            }

            for (int w = 0;w<output_w;w++)
            {
                for (int h=0;h<output_h;h++)
                {
                    *(output+out_channel*out_channel_size+h*output_w+w) = *(padded_out+out_channel*padded_out_channel_size+h*padded_out_w+w);
                }
            }
        }
        free(padded_out);
        free(padded_input);
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::forward_cpu_gemm(const Dtype *input,
                                                       const Dtype *weights, Dtype *output, bool skip_im2col) {
        const Dtype *col_buff = input;
        if (!is_1x1_) {
            if (!skip_im2col) {
                conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
            }
            col_buff = col_buffer_.cpu_data();
        }
        for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
                                                              group_, conv_out_spatial_dim_, kernel_dim_,
                                  (Dtype) 1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype) 0., output + output_offset_ * g);
        }
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::forward_cpu_bias(Dtype *output,
                                                       const Dtype *bias) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                              out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.cpu_data(),
                              (Dtype) 1., output);
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::backward_cpu_gemm(const Dtype *output,
                                                        const Dtype *weights, Dtype *input) {
        Dtype *col_buff = col_buffer_.mutable_cpu_data();
        if (is_1x1_) {
            col_buff = input;
        }
        for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                                  conv_out_spatial_dim_, conv_out_channels_ / group_,
                                  (Dtype) 1., weights + weight_offset_ * g, output + output_offset_ * g,
                                  (Dtype) 0., col_buff + col_offset_ * g);
        }
        if (!is_1x1_) {
            conv_col2im_cpu(col_buff, input);
        }
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::weight_cpu_gemm(const Dtype *input,
                                                      const Dtype *output, Dtype *weights) {
        const Dtype *col_buff = input;
        if (!is_1x1_) {
            conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
            col_buff = col_buffer_.cpu_data();
        }
        for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                                  kernel_dim_, conv_out_spatial_dim_,
                                  (Dtype) 1., output + output_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype) 1., weights + weight_offset_ * g);
        }
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::backward_cpu_bias(Dtype *bias, const Dtype *input) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                              input, bias_multiplier_.cpu_data(), 1., bias);
    }

#ifndef CPU_ONLY


    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::get_input_width(int &out)
    {
        out = conv_input_shape_.cpu_data()[2];
    }
    
    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::get_input_height(int &out)
    {
        out = conv_input_shape_.cpu_data()[1];
    }
    
    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::get_pad_width(int &out)
    {
        out = pad_.cpu_data()[1];
    }
    
    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::get_pad_height(int &out)
    {
        out = pad_.cpu_data()[0];
    }

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::get_conv_in_channels(int &out)
    {
        out = conv_in_channels_;
    }
    

    template<typename Dtype>
    void BaseWinogradLayer<Dtype>::forward_gpu_winograd(const Dtype *input, const Dtype *weights, Dtype *output) {
        
        // kernel_dim_;
        int in_channels  = conv_in_channels_;
        int out_channels = conv_out_channels_;
        int input_h      = conv_input_shape_.cpu_data()[1];
        int input_w      = conv_input_shape_.cpu_data()[2];
        int kernel_h     = kernel_shape_.cpu_data()[0];
        int kernel_w     = kernel_shape_.cpu_data()[1];
        int pad_h        = pad_.cpu_data()[0];
        int pad_w        = pad_.cpu_data()[1];
        int stride_h     = stride_.cpu_data()[0];
        int stride_w     = stride_.cpu_data()[1];
        int dilation_h   = dilation_.cpu_data()[0];
        int dilation_w   = dilation_.cpu_data()[1];
        int kernel_size  = kernel_h * kernel_w;


        if (kernel_h != 3 || kernel_w != 3) {
            LOG(FATAL) << "kernel size must be 3";
        }
        if (pad_h>4||pad_w>4)
        {
            LOG(FATAL) << "padding must less than 4";
        }
        if (group_ > 1) {
            LOG(FATAL) << "multi Groups not implemented ";
        }


        Dtype weight[3][3];  //kernel weight
        Dtype in[6][6]; //input tile
        Dtype out_tile[4][4]; //out put tile
        const int output_h = (input_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (input_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = input_h * input_w;
        const int out_channel_size = output_h*output_w;
        cudaMemset(output,0, sizeof(Dtype)*output_h*output_w*out_channels);

        // parameters of padding and tiling
        int tile_num_w = (input_w + 2 * pad_w-6) / 4 + ((input_w + 2 * pad_w-6) % 4 > 0 ? 1 : 0)+1;
        int tile_num_h = (input_h + 2 * pad_h-6) / 4 + ((input_h + 2 * pad_h-6) % 4 > 0 ? 1 : 0)+1;

        int padded_in_w  = 4*tile_num_w+2;
        int padded_in_h  = 4*tile_num_h+2;
        int padded_out_w = 4*tile_num_w;
        int padded_out_h = 4*tile_num_h;
        int padded_channel_size     = padded_in_h*padded_in_w;
        int padded_out_channel_size = padded_out_w*padded_out_h;

        // Dtype*padded_out = (Dtype*)malloc(padded_out_channel_size*out_channels* sizeof(Dtype));
        Dtype*padded_out;
        cudaError_t rc =cudaMalloc((void **)&padded_out, padded_out_channel_size*out_channels* sizeof(Dtype));
        if (rc != cudaSuccess)
        //     printf("Could not allocate memory1: %d", rc);
        // else
        //     printf("Yayyyyy1!!!! allocate memory: %d", rc);
        cudaMemset(padded_out,0, sizeof(Dtype)*padded_out_channel_size*out_channels);


        //pad 0
        // Dtype* padded_input = (Dtype*)malloc(in_channels*padded_channel_size* sizeof(Dtype));
        Dtype*padded_input;
        rc = cudaMalloc((void **)&padded_input, in_channels*padded_channel_size* sizeof(Dtype));
        // if (rc != cudaSuccess)
        //     printf("Could not allocate memory2: %d", rc);
        // else
        //     printf("Yayyyyy2!!!! allocate memory: %d", rc);
        cudaMemset(padded_input,0, sizeof(Dtype)*in_channels*padded_channel_size);

        for (int c=0;c<in_channels;c++)
            for (int h=0;h<input_h;h++)
                for (int w=0;w<input_w;w++)
                {
                     *(padded_input+c*padded_channel_size+padded_in_w*(h+pad_h)+w+pad_w)
                     = *(input+c*channel_size+h*input_w+w);
                }

        //copy input to padded_input
        // for (int c=0;c<in_channels;c++)
        //     for (int h=0;h<input_h;h++)
        //         for (int w=0;w<input_w;w++)
        //         {
        //              padded_input[c*padded_channel_size+padded_in_w*(h+pad_h)+w+pad_w] 
        //              = input[c*channel_size+h*input_w+w];
        //         }
        //         int tile_x = 0; //tile index x
        //         int tile_y = 0; //tile index y

        //         for (int out_channel = 0; out_channel < out_channels; out_channel++) {
        //             for (int tile_ind_x = 0; tile_ind_x < tile_num_w ; tile_ind_x++)
        //             {
        //                 for (int tile_ind_y = 0; tile_ind_y < tile_num_h ; tile_ind_y++) {
        //                     for (int in_channel = 0; in_channel < in_channels; in_channel++) {
        //                         for (int i = 0; i < kernel_w; i++) {
        //                             for (int j = 0; j < kernel_h; j++)
        //                             {
        //                                 // weight[i][j] = *(weights + out_channel * in_channels * kernel_size + in_channel * 3*3 + j * 3 + i);
        //                             }
        //                         }

        //                         tile_x = tile_ind_x * 4;
        //                         tile_y = tile_ind_y * 4;
        //                         //insert input tile data
        //                         for (int i = 0; i < 6; i++) {
        //                             for (int j = 0; j < 6; j++)
        //                             {
        //                                 // in[i][j] = *(padded_input + in_channel * padded_in_h*padded_in_w + (tile_y+j)*padded_in_w  + tile_x + i);
        //                             }
        //                         }

        //                         // this->winograd_4_4_3_3(weight, in, out_tile);
        //                         // this->flatten(out_tile,padded_out,tile_ind_x,tile_ind_y,out_channel,padded_out_w,padded_out_h);
        //                     }
        //                 }
        //             }

        //             for (int w = 0;w<output_w;w++)
        //             {
        //                 for (int h=0;h<output_h;h++)
        //                 {
        //                     // *(output+out_channel*out_channel_size+h*output_w+w) = *(padded_out+out_channel*padded_out_channel_size+h*padded_out_w+w);
        //                 }
        //             }
        //         }
        cudaFree(padded_out);
        cudaFree(padded_input);
    }


    template <typename Dtype>
    void BaseWinogradLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
        const Dtype* weights, Dtype* output, bool skip_im2col) {
      
      //forward_gpu_winograd(input, weights,output);
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
    void BaseWinogradLayer<Dtype>::forward_gpu_bias(Dtype* output,
        const Dtype* bias) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
          (Dtype)1., output);
    }

    template <typename Dtype>
    void BaseWinogradLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
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
    void BaseWinogradLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
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
    void BaseWinogradLayer<Dtype>::backward_gpu_bias(Dtype* bias,
        const Dtype* input) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
          input, bias_multiplier_.gpu_data(), 1., bias);
    }

#endif  // !CPU_ONLY

    INSTANTIATE_CLASS(BaseWinogradLayer);

}  // namespace caffe
