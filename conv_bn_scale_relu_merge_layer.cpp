#include "caffe/ex_layers/conv_bn_scale_relu_merge_layer.hpp"
#include <typeinfo>
#include <iostream>
using namespace std;
namespace caffe 
{
    template <typename Dtype>
    void ConvBnScaleReluMergeLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        //准备层：层参数
        LayerParameter conv_param(this->layer_param_);

        conv_param.set_type("Convolution");
        conv_param.mutable_convolution_param()->CopyFrom(
            this->layer_param_.convolution_param() );
        
        LayerParameter bn_param;//BatchNormParameter
        bn_param.set_type("BatchNorm");
        
        
        LayerParameter scale_param;
        scale_param.set_type("Scale");
        ScaleParameter sp;
        sp.mutable_filler()->set_value(1);
        sp.set_bias_term(true);
        //sp.mutable_bias_filler()->set_type("constant");
        //sp.mutable_bias_filler()->set_value(0);
        // FillerParameter sp_filler;
        //FillerParameter sp_bias_filler;
        // sp_filler.set_value(1);
        //sp_bias_filler.set_value(0);
        // sp.set_allocated_filler(&sp_filler);
        //sp.set_allocated_filler(&sp_bias_filler);
        //scale_param.set_allocated_scale_param(sp);
        scale_param.mutable_scale_param()->CopyFrom(sp);
        
        
        
        LayerParameter relu_param;
        relu_param.set_type("ReLU");
        //创建层
        conv_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_param);
        bn_layer_ = LayerRegistry<Dtype>::CreateLayer(bn_param);
        scale_layer_ = LayerRegistry<Dtype>::CreateLayer(scale_param);
        relu_layer_ = LayerRegistry<Dtype>::CreateLayer(relu_param);
        //准备top
        conv_top_vec_.clear();
        conv_top_.reset(new Blob<Dtype>());
        conv_top_vec_.push_back(conv_top_.get());
        bn_top_vec_.clear();
        bn_top_.reset(new Blob<Dtype>());
        bn_top_vec_.push_back(bn_top_.get());
        scale_top_vec_.clear();
        scale_top_.reset(new Blob<Dtype>());
        scale_top_vec_.push_back(scale_top_.get());
        //安装setup
        conv_layer_->SetUp(bottom, conv_top_vec_);
        bn_layer_->SetUp(conv_top_vec_, bn_top_vec_);
        scale_layer_->SetUp(bn_top_vec_, scale_top_vec_);
        relu_layer_->SetUp(scale_top_vec_, top);
        //其他
        
        const vector<shared_ptr<Blob<Dtype> > >& conv_blobs = conv_layer_->blobs();
        const vector<shared_ptr<Blob<Dtype> > >& bn_blobs = bn_layer_->blobs();
        const vector<shared_ptr<Blob<Dtype> > >& scale_blobs = scale_layer_->blobs();
        
        bn_blobs[2]->mutable_cpu_data()[0] = 1;
       
        size_t blobs_num = conv_blobs.size()+ bn_blobs.size() + scale_blobs.size();
        
        this->blobs_.resize(blobs_num);
        for (size_t index = 0; index < blobs_num; index++ ) 
        {
            if(index < conv_blobs.size())
            {
                this->blobs_[index].reset(new Blob<Dtype>());
                this->blobs_[index]->Reshape( conv_blobs[index]->shape() );
                this->blobs_[index]->ShareData( *(conv_blobs[index].get()) );
                this->blobs_[index]->ShareDiff( *(conv_blobs[index].get()) );
            }
             else if(index < (conv_blobs.size() + bn_blobs.size()))
             {
                 this->blobs_[index].reset(new Blob<Dtype>());
                 this->blobs_[index]->Reshape( bn_blobs[index - conv_blobs.size()]->shape() );
                 this->blobs_[index]->ShareData( *(bn_blobs[index - conv_blobs.size()].get()) );
                 this->blobs_[index]->ShareDiff( *(bn_blobs[index - conv_blobs.size()].get()) );
             }
             else
             {
                 this->blobs_[index].reset(new Blob<Dtype>());
                 this->blobs_[index]->Reshape( scale_blobs[index - conv_blobs.size() - bn_blobs.size()]->shape() );
                 this->blobs_[index]->ShareData( *(scale_blobs[index - conv_blobs.size() - bn_blobs.size()].get()) );
                 this->blobs_[index]->ShareDiff( *(scale_blobs[index - conv_blobs.size() - bn_blobs.size()].get()) );
             }
        }
    }    

    template <typename Dtype>
    void ConvBnScaleReluMergeLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        conv_layer_->Reshape(bottom, conv_top_vec_);
        bn_layer_->Reshape(conv_top_vec_, bn_top_vec_);
        scale_layer_->Reshape(bn_top_vec_, scale_top_vec_);
        relu_layer_->Reshape(scale_top_vec_, top);
    }
    
    // template <typename Dtype>
    // __global__ void MergeConvBnScale(const int channels, Dtype* conv_blobs_0_data, Dtype* conv_blobs_1_data,\
        // const Dtype* bn_blobs_0_data, const Dtype* bn_blobs_1_data, const Dtype scale, const Dtype* a, const Dtype* b)
    // {
        // CUDA_KERNEL_LOOP(k, channels)
        // {
            // Dtype mean_k = bn_blobs_0_data[k] / scale;
            // Dtype std_k = sqrt(bn_blobs_1_data[k] / scale + 1e-5);
            // conv_blobs_0_data[k] = conv_blobs_0_data[k] * a[k] / std_k;
            // conv_blobs_1_data[k] = conv_blobs_1_data[k] * a[k] / std_k - a[k] * mean_k / std_k + b[k];
        // }
    // }
    
    template <typename Dtype>
    void ConvBnScaleReluMergeLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
       printf("*********************\n");
       printf("还未实现，敬请期待，可以参考gpu代码\n");
       printf("*********************\n");
    }
    
    template <typename Dtype>
    void ConvBnScaleReluMergeLayer<Dtype>::Backward_cpu(
        const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) 
    {
        if ( propagate_down[0] == false) return;
        relu_layer_->Backward(top, propagate_down ,scale_top_vec_);
        scale_layer_->Backward(scale_top_vec_, propagate_down, bn_top_vec_);
        bn_layer_->Backward(bn_top_vec_, propagate_down, conv_top_vec_);
        conv_layer_->Backward(conv_top_vec_, propagate_down, bottom);//propagate_down, 
    }
    #ifdef CPU_ONLY
    STUB_GPU(ConvBnScaleReluMergeLayer)
    #endif
    
    INSTANTIATE_CLASS(ConvBnScaleReluMergeLayer);
    REGISTER_LAYER_CLASS(ConvBnScaleReluMerge);
}
