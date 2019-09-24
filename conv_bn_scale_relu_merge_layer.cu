#include "caffe/ex_layers/conv_bn_scale_relu_merge_layer.hpp"
using namespace std;
namespace caffe 
{
    template <typename Dtype>
    __global__ void MergeConvBnScale(const int kernel_num, Dtype* conv_blobs_0_data, Dtype* conv_blobs_1_data, const int num_per_kernel,\
        const Dtype* bn_blobs_0_data, const Dtype* bn_blobs_1_data, const Dtype* scale_ptr, const Dtype* a, const Dtype* b)
    {
        CUDA_KERNEL_LOOP(k, kernel_num)
        {
            Dtype scale = scale_ptr[0];
            //printf("%f\n",scale);
            Dtype mean_k = bn_blobs_0_data[k] / scale;
            Dtype std_k = sqrt(bn_blobs_1_data[k] / scale + 1e-5);
            for(int i=0;i<num_per_kernel;i++)
            {
                conv_blobs_0_data[i + k*num_per_kernel] = conv_blobs_0_data[i + k*num_per_kernel] * a[k] / std_k;
                
            }
            conv_blobs_1_data[k] = conv_blobs_1_data[k] * a[k] / std_k - a[k] * mean_k / std_k + b[k];
            //printf("%f %f %f %f\n",scale, mean_k, std_k,conv_blobs_1_data[k]);
        }
    }
    
    template <typename Dtype>
    void ConvBnScaleReluMergeLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        const vector<shared_ptr<Blob<Dtype> > >& conv_blobs = conv_layer_->blobs();
        const vector<shared_ptr<Blob<Dtype> > >& bn_blobs = bn_layer_->blobs();
        const vector<shared_ptr<Blob<Dtype> > >& scale_blobs = scale_layer_->blobs();
        size_t blobs_num = conv_blobs.size()+ bn_blobs.size() + scale_blobs.size();
        if (this->phase_ == TEST) 
        {
            
            for (size_t index = 0; index < blobs_num; index++ ) 
            {
                if(index < conv_blobs.size())
                {
                    conv_blobs[index]->CopyFrom(*(this->blobs_[index].get()), false, false);
                }
                else if(index < (conv_blobs.size() + bn_blobs.size()))
                {
                    bn_blobs[index - conv_blobs.size()]->CopyFrom(*(this->blobs_[index].get()), false, false);
                }
                else
                {
                    scale_blobs[index - conv_blobs.size()- bn_blobs.size()]->CopyFrom(*(this->blobs_[index].get()), false, false);
                }
            }
        }
      
        int kernel_num = conv_blobs[0]->num();//channels(); 
        const Dtype* scale_ptr = bn_blobs[2]->gpu_data();
        //const Dtype scale = scale_ptr[0];
        Dtype* bn_blobs_0_data = bn_blobs[0]->mutable_gpu_data();
        Dtype* bn_blobs_1_data = bn_blobs[1]->mutable_gpu_data();
        
        const Dtype* a = scale_blobs[0]->gpu_data();
        const Dtype* b = scale_blobs[1]->gpu_data();
        Dtype* conv_blobs_0_data = conv_blobs[0]->mutable_gpu_data();
        
        
        Dtype* conv_blobs_1_data = conv_blobs[1]->mutable_gpu_data();
        const int num_per_kernel = conv_blobs[0]->channels()* conv_blobs[0]->width() * conv_blobs[0]->height();
        
        Dtype* weight_buff = NULL;
        Dtype* bias_buff = NULL;
        int weight_count = conv_blobs[0]->count();
        int bias_count = conv_blobs[1]->count();
        cudaMalloc((void**)&weight_buff,weight_count*sizeof(Dtype));
        cudaMalloc((void**)&bias_buff,bias_count*sizeof(Dtype));
        cudaMemcpy(weight_buff, conv_blobs[0]->gpu_data(), sizeof(Dtype)*weight_count, cudaMemcpyDeviceToDevice);
        cudaMemcpy(bias_buff, conv_blobs[1]->gpu_data(), sizeof(Dtype)*bias_count, cudaMemcpyDeviceToDevice);
        
        conv_layer_->Forward(bottom, conv_top_vec_);
        bn_layer_->Forward(conv_top_vec_, bn_top_vec_);
        
        vector<Blob<Dtype>*> conv_top_vec_tmp;
        shared_ptr<Blob<Dtype> > conv_top_tmp;
        conv_top_vec_tmp.clear();
        conv_top_tmp.reset(new Blob<Dtype>());
        conv_top_tmp->ReshapeLike(*conv_top_vec_[0]);
        cudaMemcpy(conv_top_tmp->mutable_gpu_data(), conv_top_vec_[0]->gpu_data(), sizeof(Dtype)*conv_top_vec_[0]->count(), cudaMemcpyDeviceToDevice);
        conv_top_vec_tmp.push_back(conv_top_tmp.get());
        
        MergeConvBnScale <<<CAFFE_GET_BLOCKS(kernel_num), CAFFE_CUDA_NUM_THREADS >>> \
           (kernel_num, conv_blobs_0_data, conv_blobs_1_data, num_per_kernel, bn_blobs_0_data, bn_blobs_1_data, scale_ptr, a, b);
        conv_layer_->Forward(bottom, conv_top_vec_);
        cudaMemcpy(scale_top_vec_[0]->mutable_gpu_data(), conv_top_vec_[0]->gpu_data(), sizeof(Dtype)*conv_top_vec_[0]->count(), cudaMemcpyDeviceToDevice);
        relu_layer_->Forward(scale_top_vec_, top);
        
        cudaMemcpy(conv_top_vec_[0]->mutable_gpu_data(),conv_top_tmp->gpu_data(), sizeof(Dtype)*conv_top_vec_[0]->count(), cudaMemcpyDeviceToDevice);
        cudaMemcpy(conv_blobs[0]->mutable_gpu_data(), weight_buff, sizeof(Dtype)*weight_count, cudaMemcpyDeviceToDevice);
        cudaMemcpy(conv_blobs[1]->mutable_gpu_data(), bias_buff, sizeof(Dtype)*bias_count, cudaMemcpyDeviceToDevice);
        cudaFree(weight_buff);
        cudaFree(bias_buff);
        
        
        //Dtype* top_buff = (Dtype*)malloc(top[0]->count()*sizeof(Dtype));
        //cudaMemcpy(top_buff, top[0]->gpu_data(), sizeof(Dtype)*top[0]->count(), cudaMemcpyDeviceToHost);
        
        // conv_layer_->Forward(bottom, conv_top_vec_);
        // bn_layer_->Forward(conv_top_vec_, bn_top_vec_);
        // scale_layer_->Forward(bn_top_vec_, scale_top_vec_);
        // relu_layer_->Forward(scale_top_vec_, top);
        // for(int i=0;i<top[0]->count();i++)
        // {
           // CHECK_EQ(top[0]->cpu_data()[i], top_buff[i]);
        // }
        
    }
    
    template <typename Dtype>
    void ConvBnScaleReluMergeLayer<Dtype>::Backward_gpu(
        const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) 
    {
        if ( propagate_down[0] == false) return;
        relu_layer_->Backward(top, propagate_down ,scale_top_vec_);
        scale_layer_->Backward(scale_top_vec_, propagate_down, bn_top_vec_);
        bn_layer_->Backward(bn_top_vec_, propagate_down, conv_top_vec_);
        conv_layer_->Backward(conv_top_vec_, propagate_down, bottom);//propagate_down, 
    }
    INSTANTIATE_LAYER_GPU_FUNCS(ConvBnScaleReluMergeLayer);
}