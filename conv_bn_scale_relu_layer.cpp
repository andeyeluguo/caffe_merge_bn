#include "caffe/ex_layers/conv_bn_scale_relu_layer.hpp"
#include <typeinfo>
#include <iostream>
using namespace std;
namespace caffe 
{
    template <typename Dtype>
    void ConvBnScaleReluLayer<Dtype>::LayerSetUp(
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
    void ConvBnScaleReluLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        conv_layer_->Reshape(bottom, conv_top_vec_);
        bn_layer_->Reshape(conv_top_vec_, bn_top_vec_);
        scale_layer_->Reshape(bn_top_vec_, scale_top_vec_);
        relu_layer_->Reshape(scale_top_vec_, top);
    }
    
    template <typename Dtype>
    void ConvBnScaleReluLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
        if (this->phase_ == TEST) 
        {
            const vector<shared_ptr<Blob<Dtype> > >& conv_blobs = conv_layer_->blobs();
            const vector<shared_ptr<Blob<Dtype> > >& bn_blobs = bn_layer_->blobs();
            const vector<shared_ptr<Blob<Dtype> > >& scale_blobs = scale_layer_->blobs();
            size_t blobs_num = conv_blobs.size()+ bn_blobs.size() + scale_blobs.size();
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
        
        
        conv_layer_->Forward(bottom, conv_top_vec_);
        bn_layer_->Forward(conv_top_vec_, bn_top_vec_);
        scale_layer_->Forward(bn_top_vec_, scale_top_vec_);
        relu_layer_->Forward(scale_top_vec_, top);
        //printf("********************************\n");
        // printf("%d \n", this->blobs_.size());
        //printf("%d %d %d %d %d\n",conv_top_vec_[0]->shape(1),bn_top_vec_[0]->shape(1),bn_top_vec_[0]->shape(1),scale_top_vec_[0]->shape(1),top[0]->shape(1));
        //printf("********************************\n");
    }
    
    template <typename Dtype>
    void ConvBnScaleReluLayer<Dtype>::Backward_cpu(
        const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) 
    {
        if ( propagate_down[0] == false) return;
        // printf("*************************");
        // printf("已反向\n");
        // printf("*************************");
        relu_layer_->Backward(top, propagate_down ,scale_top_vec_);
        scale_layer_->Backward(scale_top_vec_, propagate_down, bn_top_vec_);
        bn_layer_->Backward(bn_top_vec_, propagate_down, conv_top_vec_);
        conv_layer_->Backward(conv_top_vec_, propagate_down, bottom);//propagate_down, 
    }
    
    INSTANTIATE_CLASS(ConvBnScaleReluLayer);
    REGISTER_LAYER_CLASS(ConvBnScaleRelu);
}
