#include <vector>

#include "caffe/layers/wino_layer.hpp"

namespace caffe {

    template<typename Dtype>
    void WinogradLayer<Dtype>::compute_output_shape() {
        const int *kernel_shape_data = this->kernel_shape_.gpu_data();
        const int *stride_data = this->stride_.gpu_data();
        const int *pad_data = this->pad_.gpu_data();
        const int *dilation_data = this->dilation_.gpu_data();
        this->output_shape_.clear();
        for (int i = 0; i < this->num_spatial_axes_; ++i) {
            // i + 1 to skip channel axis
            const int input_dim = this->input_shape(i + 1);
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
            const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                                   / stride_data[i] + 1;
            this->output_shape_.push_back(output_dim);
        }
    }

    template<typename Dtype>
    void WinogradLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {
        const Dtype *weight = this->blobs_[0]->gpu_data();
        for (int i = 0; i < bottom.size(); ++i) {
            const Dtype *bottom_data = bottom[i]->gpu_data();
            Dtype *top_data = top[i]->mutable_gpu_data();
            for (int n = 0; n < this->num_; ++n) {
                const int *kernel_shape_data = this->kernel_shape_.gpu_data();
                if (kernel_shape_data[i] < 3) //kernel size !=3 has not implemented
                    this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                           top_data + n * this->top_dim_);
                else {
                    this->forward_gpu_winograd(bottom_data + n * this->bottom_dim_, weight,
                                               top_data + n * this->top_dim_);
                    //this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                    //                       top_data + n * this->top_dim_);
                }

                if (this->bias_term_) {
                    const Dtype *bias = this->blobs_[1]->gpu_data();
                    this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
                }
            }
        }
    }



    template<typename Dtype>
    void forward_gpu_winograd(const Dtype *input, const Dtype *weights, Dtype *output) {
        kernel_dim_;
        int in_channels  = conv_in_channels_;
        int out_channels = conv_out_channels_;
        int input_h      = conv_input_shape_.gpu_data()[1];
        int input_w      = conv_input_shape_.gpu_data()[2];
        int kernel_h     = kernel_shape_.gpu_data()[0];
        int kernel_w     = kernel_shape_.gpu_data()[1];
        int pad_h        = pad_.gpu_data()[0];
        int pad_w        = pad_.gpu_data()[1];
        int stride_h     = stride_.gpu_data()[0];
        int stride_w     = stride_.gpu_data()[1];
        int dilation_h   = dilation_.gpu_data()[0];
        int dilation_w   = dilation_.gpu_data()[1];
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
                        for (int i = 0; i < kernel_w; i++) {
                            for (int j = 0; j < kernel_h; j++)
                            {
                                weight[i][j] = *(weights + out_channel * in_channels * kernel_size + in_channel * 3*3 + j * 3 + i);
                            }
                        }

                        tile_x = tile_ind_x * 4;
                        tile_y = tile_ind_y * 4;
                        //insert input tile data
                        for (int i = 0; i < 6; i++) {
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
    void WinogradLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                               const vector<bool> &propagate_down,
                                               const vector<Blob<Dtype> *> &bottom) {
        const Dtype *weight = this->blobs_[0]->gpu_data();
        Dtype *weight_diff = this->blobs_[0]->mutable_gpu_diff();
        for (int i = 0; i < top.size(); ++i) {
            const Dtype *top_diff = top[i]->gpu_diff();
            const Dtype *bottom_data = bottom[i]->gpu_data();
            Dtype *bottom_diff = bottom[i]->mutable_gpu_diff();
            // Bias gradient, if necessary.
            if (this->bias_term_ && this->param_propagate_down_[1]) {
                Dtype *bias_diff = this->blobs_[1]->mutable_gpu_diff();
                for (int n = 0; n < this->num_; ++n) {
                    this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
                }
            }
            if (this->param_propagate_down_[0] || propagate_down[i]) {
                for (int n = 0; n < this->num_; ++n) {
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0]) {
                        this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                                              top_diff + n * this->top_dim_, weight_diff);
                    }
                    // gradient w.r.t. bottom data, if necessary.
                    if (propagate_down[i]) {
                        this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                                                bottom_diff + n * this->bottom_dim_);
                    }
                }
            }
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(WinogradLayer);

}  // namespace caffe
