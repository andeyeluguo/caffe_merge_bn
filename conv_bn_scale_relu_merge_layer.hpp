#ifndef CAFFE_ConvBnScaleReluMerge_LAYER_HPP_
#define CAFFE_ConvBnScaleReluMerge_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe
{
    template <typename Dtype>
    class ConvBnScaleReluMergeLayer : public Layer<Dtype>
    {
        public:
              explicit ConvBnScaleReluMergeLayer(const LayerParameter& param)
                    : Layer<Dtype>(param) {}

            virtual inline const char* type() const { return "ConvBnScaleReluMergeLayer"; }
        
            virtual void LayerSetUp(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
            virtual  void Reshape(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top); 
        protected:
            virtual void Forward_cpu(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
            virtual void Backward_cpu(
                const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                const vector<Blob<Dtype>*>& bottom) ;
            virtual void Forward_gpu(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
            virtual void Backward_gpu(
                const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                const vector<Blob<Dtype>*>& bottom) ;
            
            shared_ptr<Layer<Dtype> > conv_layer_;
            shared_ptr<Layer<Dtype> > bn_layer_;
            shared_ptr<Layer<Dtype> > scale_layer_;
            shared_ptr<Layer<Dtype> > relu_layer_;
            
            shared_ptr<Blob<Dtype> > conv_top_;
            shared_ptr<Blob<Dtype> > bn_top_;
            shared_ptr<Blob<Dtype> > scale_top_;
            vector<Blob<Dtype>*> conv_top_vec_;
            vector<Blob<Dtype>*> bn_top_vec_;
            vector<Blob<Dtype>*> scale_top_vec_;
            
            // virtual inline void copyGPUTo(const shared_ptr<Blob<Dtype> > ori, const shared_ptr<Blob<Dtype> > buf) 
            // {
                // CHECK_EQ(ori->count(), buf->count());
                // cudaMemcpy(buf->mutable_gpu_data(), ori->gpu_data(), sizeof(Dtype)*ori->count(), cudaMemcpyDefault);
            // }
            
            // virtual inline void copyGPUTo_1(const Blob<Dtype>* ori, Blob<Dtype>* buf) 
            // {
                // CHECK_EQ(ori->count(), buf->count());
                // cudaMemcpy(buf->mutable_gpu_data(), ori->gpu_data(), sizeof(Dtype)*ori->count(), cudaMemcpyDeviceToDevice);
            // }
        
            
    };
}
#endif 