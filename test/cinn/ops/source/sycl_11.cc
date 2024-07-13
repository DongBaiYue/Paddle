#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_arange_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  float*  var = (float* )(*(void **)(void_args[0]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space11_fn_arange_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(6, 1, 1)]]
    {
      if (((int)item.get_local_id(2) < 6)) {
        var[(int)item.get_local_id(2)] = (1024.00000f + (512.000000f * ((float)((int)item.get_local_id(2)))));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
