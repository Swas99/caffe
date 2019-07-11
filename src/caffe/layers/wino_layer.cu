#include <stdio.h>
#include <vector>

#include "caffe/layers/wino_layer.hpp"

namespace caffe {


    // dim3 threadsPerBlock(C)
    // dim3 numBlocks(Batch, nH, nW)

    // I = (Batch, H, W, C)
    // O = (16, Batch, nH, nW, C)
    template <typename T>
    __global__ void Winograd2x2ImTransCompute(const T *Input, T *Output, int C, int B, int H, int W, int pad_h, int pad_w)
    { 
        int bx = blockIdx.x; // w
        int by = blockIdx.y; // h
        int bz = blockIdx.z; // b 
        int t = threadIdx.x; // c

        int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
        int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;

        int f_b = bz;
        int xBase = 2 * bx - pad_w;
        int yBase = 2 * by - pad_h;

        // T input_patch_1 [16] = {0};
        T input_patch_0;
        T input_patch_1;
        T input_patch_2;
        T input_patch_3;
        T input_patch_4;
        T input_patch_5;
        T input_patch_6;
        T input_patch_7;
        T input_patch_8;
        T input_patch_9;
        T input_patch_10;
        T input_patch_11;
        T input_patch_12;
        T input_patch_13;
        T input_patch_14;
        T input_patch_15;

        // load (4, 4, 1) patch of input from global memory
        int f_x, f_y;
        f_x = xBase + 0; f_y = yBase + 0;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_0 = 0;
        f_x = xBase + 1; f_y = yBase + 0;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_1 = 0;
        f_x = xBase + 2; f_y = yBase + 0;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_2 = 0;
        f_x = xBase + 3; f_y = yBase + 0;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_3 = 0;
        f_x = xBase + 0; f_y = yBase + 1;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_4 = 0;
        f_x = xBase + 1; f_y = yBase + 1;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_5 = 0;
        f_x = xBase + 2; f_y = yBase + 1;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_6 = 0;
        f_x = xBase + 3; f_y = yBase + 1;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_7 = 0;
        f_x = xBase + 0; f_y = yBase + 2;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_8 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_8 = 0;
        f_x = xBase + 1; f_y = yBase + 2;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_9 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_9 = 0;
        f_x = xBase + 2; f_y = yBase + 2;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_10 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_10 = 0;
        f_x = xBase + 3; f_y = yBase + 2;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_11 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_11 = 0;
        f_x = xBase + 0; f_y = yBase + 3;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_12 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_12 = 0;
        f_x = xBase + 1; f_y = yBase + 3;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_13 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_13 = 0;
        f_x = xBase + 2; f_y = yBase + 3;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_14 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_14 = 0;
        f_x = xBase + 3; f_y = yBase + 3;
        if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_15 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
        else input_patch_15 = 0;
        
        T trans_input_patch_0;
        T trans_input_patch_1;
        T trans_input_patch_2;
        T trans_input_patch_3;
        T trans_input_patch_4;
        T trans_input_patch_5;
        T trans_input_patch_6;
        T trans_input_patch_7;
        T trans_input_patch_8;
        T trans_input_patch_9;
        T trans_input_patch_10;
        T trans_input_patch_11;
        T trans_input_patch_12;
        T trans_input_patch_13;
        T trans_input_patch_14;
        T trans_input_patch_15;

        // Winograd Transform
        trans_input_patch_0 = input_patch_0 - input_patch_2 - input_patch_8 + input_patch_10;
        trans_input_patch_1 = input_patch_1 + input_patch_2 - input_patch_9 - input_patch_10;
        trans_input_patch_2 = input_patch_2 - input_patch_1 + input_patch_9 - input_patch_10;
        trans_input_patch_3 = input_patch_1 - input_patch_3 - input_patch_9 + input_patch_11;
        trans_input_patch_4 = input_patch_4 - input_patch_6 + input_patch_8 - input_patch_10;
        trans_input_patch_5 = input_patch_5 + input_patch_6 + input_patch_9 + input_patch_10;
        trans_input_patch_6 = input_patch_6 - input_patch_5 - input_patch_9 + input_patch_10;
        trans_input_patch_7 = input_patch_5 - input_patch_7 + input_patch_9 - input_patch_11;
        trans_input_patch_8 = input_patch_6 - input_patch_4 + input_patch_8 - input_patch_10;
        trans_input_patch_9 = input_patch_9 - input_patch_6 - input_patch_5 + input_patch_10;
        trans_input_patch_10 = input_patch_5 - input_patch_6 - input_patch_9 + input_patch_10;
        trans_input_patch_11 = input_patch_7 - input_patch_5 + input_patch_9 - input_patch_11;
        trans_input_patch_12 = input_patch_4 - input_patch_6 - input_patch_12 + input_patch_14;
        trans_input_patch_13 = input_patch_5 + input_patch_6 - input_patch_13 - input_patch_14;
        trans_input_patch_14 = input_patch_6 - input_patch_5 + input_patch_13 - input_patch_14;
        trans_input_patch_15 = input_patch_5 - input_patch_7 - input_patch_13 + input_patch_15;


        int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
        int stride = B * nH * nW * C;
        
        Output [ 0 * stride + offset ] = trans_input_patch_0;
        Output [ 1 * stride + offset ] = trans_input_patch_1;
        Output [ 2 * stride + offset ] = trans_input_patch_2;
        Output [ 3 * stride + offset ] = trans_input_patch_3;
        Output [ 4 * stride + offset ] = trans_input_patch_4;
        Output [ 5 * stride + offset ] = trans_input_patch_5;
        Output [ 6 * stride + offset ] = trans_input_patch_6;
        Output [ 7 * stride + offset ] = trans_input_patch_7;
        Output [ 8 * stride + offset ] = trans_input_patch_8;
        Output [ 9 * stride + offset ] = trans_input_patch_9;
        Output [ 10* stride + offset ] = trans_input_patch_10;
        Output [ 11* stride + offset ] = trans_input_patch_11;
        Output [ 12* stride + offset ] = trans_input_patch_12;
        Output [ 13* stride + offset ] = trans_input_patch_13;
        Output [ 14* stride + offset ] = trans_input_patch_14;
        Output [ 15* stride + offset ] = trans_input_patch_15;
    } 

    template<typename Dtype>
    void Winograd2x2ImTransComputeLauncher(const Dtype *Input, Dtype *TransIm, int C, int B, int H, int W, int pad_h, int pad_w) {
        int n_patch_width = (W + 1 + 2 * pad_w - 4) / 2 + 1;
        int n_patch_height = (H + 1 + 2 * pad_h - 4) / 2 + 1;
        dim3 blockDim(C, 1, 1);
        dim3 gridDim(n_patch_width, n_patch_height, B);
        Winograd2x2ImTransCompute<Dtype><<<gridDim, blockDim>>>(Input, TransIm, C, B, H, W, pad_h, pad_w);
    }


    template<typename Dtype>
    void xxx(const Dtype *input, const Dtype *weights, Dtype *output, int B,int H,int W,int pad_h,int pad_w, int C) {
         
        // kernel_dim_; 

        int n_patch_width = (W + 1) / 2;
        int n_patch_height = (H + 1) / 2;
        Dtype *wTransInput;
        cudaMalloc((void **)&wTransInput, 16* B* n_patch_height * n_patch_width * C* sizeof(Dtype));
        cudaMemset(wTransInput,0, 16* B* n_patch_height * n_patch_width * C* sizeof(Dtype));
        
        Winograd2x2ImTransComputeLauncher(input, wTransInput, C, B, H, W,1,1);
    }

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


            int H,W,pad_h,pad_w,C;
            this->get_input_height(H);
            this->get_input_width(W);
            this->get_pad_height(pad_h);
            this->get_pad_width(pad_w);
            this->get_conv_in_channels(C);
            //printf("B: %d \n", this->num_);
            //printf("C: %d \n", C);
            //printf("input_h: %d \n", H);
            //printf("input_w: %d \n", W);
            //printf("pad_h: %d \n", pad_h);
            //printf("pad_w: %d \n", pad_w);
            xxx(bottom_data, weight, top_data, this->num_,H,W,pad_h,pad_w,C);

            const int *kernel_shape_data = this->kernel_shape_.gpu_data();
            for (int n = 0; n < this->num_; ++n) {
                if (kernel_shape_data[i] < 3) //kernel size !=3 has not implemented
                    this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                           top_data + n * this->top_dim_);
                else {
                    //this->forward_gpu_winograd(bottom_data + n * this->bottom_dim_, weight,
                    //                           top_data + n * this->top_dim_);
                    this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                           top_data + n * this->top_dim_);
                }

                if (this->bias_term_) {
                    const Dtype *bias = this->blobs_[1]->gpu_data();
                    this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
                }
            }
        }
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
