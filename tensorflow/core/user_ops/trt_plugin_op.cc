#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// The plugin op registered as the placeholder of TRT plugin.
// No real kernel implementation will be provided for this op
// 
REGISTER_OP("TrtPluginOp")
    .Input("test_in: float")
    .Output("test_out: float")
    .Attr("trt_plugin_name: string")
    .Attr("trt_plugin_attrs: list(func)");              // List of attributes


// Kernel for op
// This kernel implementation will return 0 tensor for any input

class TRTPluginOpKernel : public OpKernel {
 public:
  // In the construction of node may require initialize the attributes from 
  explicit TRTPluginOpKernel(OpKernelConstruction* context) : OpKernel(context) {}

  // Output size cannot be decided. Return zero tensor the same size as input(0)
  void Compute(OpKernelContext *context) override {
    const Tensor input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    auto output_flat = output_tensor->flat<int32>();

    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("TRTPluginOp").Device(DEVICE_CPU), TRTPluginOpKernel);

}


#endif  //GOOGLE_TENSORRT
#endif  //GOOGLE_CUDA
                                                                                                      