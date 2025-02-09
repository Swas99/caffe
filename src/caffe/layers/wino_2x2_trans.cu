#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/wino_2x2_trans.hpp"

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

    
    void Winograd2x2ImTransComputeLauncher(const float *Input, float *TransIm, int C, int B, int H, int W, int pad_h, int pad_w) {
        int n_patch_width = (W + 1 + 2 * pad_w - 4) / 2 + 1;
        int n_patch_height = (H + 1 + 2 * pad_h - 4) / 2 + 1;
        dim3 blockDim(C, 1, 1);
        dim3 gridDim(n_patch_width, n_patch_height, B);
        Winograd2x2ImTransCompute<float><<<gridDim, blockDim>>>(Input, TransIm, C, B, H, W, pad_h, pad_w);
    }



    void WinogradTransform(const float *input, const float *weights, float *output, int B,int H,int W,int pad_h,int pad_w, int C, int K) {
         
        // kernel_dim_; 

        int nW = (W + 1) / 2;
        int nH = (H + 1) / 2;
        float *wTransInput;
        cudaMalloc((void **)&wTransInput, 16* B* nH * nW * C* sizeof(float));
        cudaMemset(wTransInput,0, 16* B* nH * nW * C* sizeof(float));
        
        Winograd2x2ImTransComputeLauncher(input, wTransInput, C, B, H, W,1,1);
    }


    void WinogradTransform(const double *input, const double *weights, double *output, int B,int H,int W,int pad_h,int pad_w, int C, int K) {
         
    }

    

    template<typename Dtype>
    void Winograd2x2TransLayer<Dtype>::compute_output_shape() {
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
    void Winograd2x2TransLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
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
            const int *kernel_shape_data = this->kernel_shape_.cpu_data();

            //printf("B: %d \n", this->num_);
            //printf("C: %d \n", C);
            //printf("input_h: %d \n", H);
            //printf("input_w: %d \n", W);
            //printf("pad_h: %d \n", pad_h);
            //printf("pad_w: %d \n", pad_w);
            //printf("K: %d \n", kernel_shape_data[i]);
            WinogradTransform(bottom_data, weight, top_data, this->num_,H,W,pad_h,pad_w,C,kernel_shape_data[i]);
        }
    }

    
    template<typename Dtype>
    void Winograd2x2TransLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                               const vector<bool> &propagate_down,
                                               const vector<Blob<Dtype> *> &bottom) {

       
    }


    void WinogradGradientTransform(const float *input, const float *weights, float *output, int B,int H,int W,int pad_h,int pad_w, int C, int K) {
         
        // kernel_dim_; 

        int nW = (W + 1) / 2;
        int nH = (H + 1) / 2;
        float *wTransInput;
        cudaMalloc((void **)&wTransInput, 16* B* nH * nW * C* sizeof(float));
        cudaMemset(wTransInput,0, 16* B* nH * nW * C* sizeof(float));
        
        Winograd2x2ImTransComputeLauncher(input, wTransInput, C, B, H, W,1,1);
    }


    void WinogradGradientTransform(const double *input, const double *weights, double *output, int B,int H,int W,int pad_h,int pad_w, int C, int K) {
         
    }


// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// O = (16, Batch, nH, nW, C)
template <typename T>
__global__ void OutputGradTransform(float *Output_grad, int C, int B, int H, int W, int pad_h, int pad_w) {
    int bx = blockIdx.x; // nw
    int by = blockIdx.y; // nh
    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // c

    int nH = (H + 1) / 2;
    int nW = (W + 1) / 2;

    int offset_1 = bz * nH * nW * C + (by * nW + bx) * C + tx;
    int stride_1 = B * nH * nW * C;

    T trans_input_grad_patch_0 = Output_grad [ 0 * stride_1 + offset_1 ];
    T trans_input_grad_patch_1 = Output_grad [ 1 * stride_1 + offset_1 ];
    T trans_input_grad_patch_2 = Output_grad [ 2 * stride_1 + offset_1 ];
    T trans_input_grad_patch_3 = Output_grad [ 3 * stride_1 + offset_1 ];
    T trans_input_grad_patch_4 = Output_grad [ 4 * stride_1 + offset_1 ];
    T trans_input_grad_patch_5 = Output_grad [ 5 * stride_1 + offset_1 ];
    T trans_input_grad_patch_6 = Output_grad [ 6 * stride_1 + offset_1 ];
    T trans_input_grad_patch_7 = Output_grad [ 7 * stride_1 + offset_1 ];
    T trans_input_grad_patch_8 = Output_grad [ 8 * stride_1 + offset_1 ];
    T trans_input_grad_patch_9 = Output_grad [ 9 * stride_1 + offset_1 ];
    T trans_input_grad_patch_10= Output_grad [ 10* stride_1 + offset_1 ];
    T trans_input_grad_patch_11= Output_grad [ 11* stride_1 + offset_1 ];
    T trans_input_grad_patch_12= Output_grad [ 12* stride_1 + offset_1 ];
    T trans_input_grad_patch_13= Output_grad [ 13* stride_1 + offset_1 ];
    T trans_input_grad_patch_14= Output_grad [ 14* stride_1 + offset_1 ];
    T trans_input_grad_patch_15= Output_grad [ 15* stride_1 + offset_1 ];

    T input_grad_patch_0 = trans_input_grad_patch_0; 
    T input_grad_patch_1 = trans_input_grad_patch_1 - trans_input_grad_patch_2 + trans_input_grad_patch_3;
    T input_grad_patch_2 = trans_input_grad_patch_1 - trans_input_grad_patch_0 + trans_input_grad_patch_2;     
    T input_grad_patch_3 =-trans_input_grad_patch_3;
    T input_grad_patch_4 = trans_input_grad_patch_4 - trans_input_grad_patch_8 + trans_input_grad_patch_12; 
    T input_grad_patch_5 = trans_input_grad_patch_5 - trans_input_grad_patch_6 + trans_input_grad_patch_7 - 
                                     trans_input_grad_patch_9 + trans_input_grad_patch_10 - trans_input_grad_patch_11 + 
                                     trans_input_grad_patch_13 - trans_input_grad_patch_14 + trans_input_grad_patch_15; 
    T input_grad_patch_6 = trans_input_grad_patch_5 - trans_input_grad_patch_4 + trans_input_grad_patch_6 + 
                                     trans_input_grad_patch_8 - trans_input_grad_patch_9 - trans_input_grad_patch_10 - 
                                     trans_input_grad_patch_12 + trans_input_grad_patch_13 + trans_input_grad_patch_14; 
    T input_grad_patch_7 = trans_input_grad_patch_11 - trans_input_grad_patch_7 - trans_input_grad_patch_15;
    T input_grad_patch_8 = trans_input_grad_patch_4 - trans_input_grad_patch_0 + trans_input_grad_patch_8; 
    T input_grad_patch_9 = trans_input_grad_patch_2 - trans_input_grad_patch_1 - trans_input_grad_patch_3 + 
                                     trans_input_grad_patch_5 - trans_input_grad_patch_6 + trans_input_grad_patch_7 + 
                                     trans_input_grad_patch_9 - trans_input_grad_patch_10 + trans_input_grad_patch_11;    
    T input_grad_patch_10= trans_input_grad_patch_0 - trans_input_grad_patch_1 - trans_input_grad_patch_2 - 
                                     trans_input_grad_patch_4 + trans_input_grad_patch_5 + trans_input_grad_patch_6 - 
                                     trans_input_grad_patch_8 + trans_input_grad_patch_9 + trans_input_grad_patch_10;  
    T input_grad_patch_11= trans_input_grad_patch_3 - trans_input_grad_patch_7 - trans_input_grad_patch_11;
    T input_grad_patch_12=-trans_input_grad_patch_12;
    T input_grad_patch_13= trans_input_grad_patch_14 - trans_input_grad_patch_13 - trans_input_grad_patch_15;  
    T input_grad_patch_14= trans_input_grad_patch_12 - trans_input_grad_patch_13 - trans_input_grad_patch_14;
    T input_grad_patch_15= trans_input_grad_patch_15;

    __syncthreads();
    Output_grad [ 0 * stride_1 + offset_1 ] = input_grad_patch_0;
    Output_grad [ 1 * stride_1 + offset_1 ] = input_grad_patch_1;
    Output_grad [ 2 * stride_1 + offset_1 ] = input_grad_patch_2;
    Output_grad [ 3 * stride_1 + offset_1 ] = input_grad_patch_3;
    Output_grad [ 4 * stride_1 + offset_1 ] = input_grad_patch_4;
    Output_grad [ 5 * stride_1 + offset_1 ] = input_grad_patch_5;
    Output_grad [ 6 * stride_1 + offset_1 ] = input_grad_patch_6;
    Output_grad [ 7 * stride_1 + offset_1 ] = input_grad_patch_7;
    Output_grad [ 8 * stride_1 + offset_1 ] = input_grad_patch_8;
    Output_grad [ 9 * stride_1 + offset_1 ] = input_grad_patch_9;
    Output_grad [ 10* stride_1 + offset_1 ] = input_grad_patch_10;
    Output_grad [ 11* stride_1 + offset_1 ] = input_grad_patch_11;
    Output_grad [ 12* stride_1 + offset_1 ] = input_grad_patch_12;
    Output_grad [ 13* stride_1 + offset_1 ] = input_grad_patch_13;
    Output_grad [ 14* stride_1 + offset_1 ] = input_grad_patch_14;
    Output_grad [ 15* stride_1 + offset_1 ] = input_grad_patch_15;
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, H, W)

// I = (Batch, H, W, C)
// O = (16, Batch, nH, nW, C)
template <typename T>
__global__ void Winograd2x2ImTransGradCompute(const float *Output_grad, float *Input_grad, int C, int B, int H, int W, int pad_h, int pad_w) {
    int bx = blockIdx.x; // w
    int by = blockIdx.y; // h
    int bz = blockIdx.z; // b
    int tx = threadIdx.x; // c

    int nH = (H + 1) / 2;
    int nW = (W + 1) / 2;

    int w_eff = bx + pad_w;
    int h_eff = by + pad_h;
    int w_col_start = (w_eff < 4) ? 0 : (w_eff - 4) / 2 + 1;
    int w_col_end = min(w_eff / 2 + 1, nW);
    int h_col_start = (h_eff < 4) ? 0 : (h_eff - 4) / 2 + 1;
    int h_col_end = min(h_eff / 2 + 1, nH);

    T val = 0;
    int offset = bz * nH * nW * C + tx;
    int stride = B * nH * nW * C;
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            int w_offset = w_eff - w_col * 2;   // within 16
            int h_offset = h_eff - h_col * 2;   // within 16
            val += Output_grad [offset + (h_offset * 4 + w_offset) * stride + (h_col * nW + w_col) * C];
        }
    }
    Input_grad[bz * H * W * C + by * W * C + bx * C + tx] = val;
} 

void Winograd2x2ImTransGradComputeLauncher(const float *Output_grad, float *Input_grad, int C, int B, int H, int W, int pad_h, int pad_w) {
    int n_patch_width = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    int n_patch_height = (H + 1 + 2 * pad_h - 4) / 2 + 1;

    // cudaMemset(Input_grad, 0, sizeof(float) * B * C * H * W); 

    OutputGradTransform<float><<<dim3(n_patch_width, n_patch_height, B), dim3(C, 1, 1)>>>((float*)Output_grad, C, B, H, W, pad_h, pad_w);

    // dim3 blockDim1(C, 1, 1);
    // dim3 gridDim1(n_patch_height, n_patch_width, B);
    Winograd2x2ImTransGradCompute<float><<<dim3(W, H, B), dim3(C, 1, 1)>>>(Output_grad, Input_grad, C, B, H, W, pad_h, pad_w);
}


    INSTANTIATE_LAYER_GPU_FUNCS(Winograd2x2TransLayer);

}  // namespace caffe