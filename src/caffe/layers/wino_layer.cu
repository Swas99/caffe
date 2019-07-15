#include <stdio.h>
#include <vector>
#include <cublas_v2.h>

#include "caffe/layers/wino_layer.hpp"

namespace caffe {

    

    // void xxx(const float *input, const float *weights, float *output, int B,int H,int W,int pad_h,int pad_w, int C, int K) {
         
    //     // kernel_dim_; 

    //     int nW = (W + 1) / 2;
    //     int nH = (H + 1) / 2;
    //     float *wTransInput;
    //     cudaMalloc((void **)&wTransInput, 16* B* nH * nW * C* sizeof(float));
    //     cudaMemset(wTransInput,0, 16* B* nH * nW * C* sizeof(float));
        
    //     Winograd2x2ImTransComputeLauncher(input, wTransInput, C, B, H, W,1,1);


    //     cudaMalloc((void **)&output, B* 2*nH * 2*nW * K * sizeof(float));
    //     cudaMemset(output,0, B* 2*nH * 2*nW * K * sizeof(float));    

    //     // Allocate temporary memory
    //     float *tmp_data_buffer_tensor;
    //     cudaMalloc((void **)&tmp_data_buffer_tensor, 16 * nH * nW * B * K * sizeof(float));
        
    //     long long *tmp_ptr_buffer_tensor;
    //     cudaMalloc((void **)&tmp_ptr_buffer_tensor, 3 * 16 * sizeof(long long));


    //     // Set all but the first element of the output tensor to 0.
    //     Winograd2x2ConvComputeLauncher(wTransInput, weights, output, 
    //     tmp_data_buffer_tensor, tmp_ptr_buffer_tensor, C, B, nH, nW, K, 1, 1); 

    //     cudaFree(wTransInput);
    //     cudaFree(tmp_ptr_buffer_tensor);
    //     cudaFree(tmp_data_buffer_tensor);
    
    // }


    void xxx(const double *input, const double *weights, double *output, int B,int H,int W,int pad_h,int pad_w, int C, int K) {
         
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


            //int H,W,pad_h,pad_w,C;
            //this->get_input_height(H);
            //this->get_input_width(W);
            //this->get_pad_height(pad_h);
            //this->get_pad_width(pad_w);
            //this->get_conv_in_channels(C);
            const int *kernel_shape_data = this->kernel_shape_.cpu_data();

            //printf("B: %d \n", this->num_);
            //printf("C: %d \n", C);
            //printf("input_h: %d \n", H);
            //printf("input_w: %d \n", W);
            //printf("pad_h: %d \n", pad_h);
            //printf("pad_w: %d \n", pad_w);
            //printf("K: %d \n", kernel_shape_data[i]);
            //xxx(bottom_data, weight, top_data, this->num_,H,W,pad_h,pad_w,C,kernel_shape_data[i]);

            for (int n = 0; n < this->num_; ++n) {
                //printf("K: %d \n", kernel_shape_data[i]);
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
                        //this->forward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                        //                        bottom_diff + n * this->bottom_dim_);
                    }
                }
            }
        }
    }



    INSTANTIATE_LAYER_GPU_FUNCS(WinogradLayer);

}  // namespace caffe
