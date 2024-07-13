#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_tan_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  float*  var_0 = (float* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space2_fn_tan_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1024, 1, 1)]]
    {
      if (((int)item.get_group(2) < 2)) {
        if (((int)item.get_local_id(2) < 1024)) {
          var_0[((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2))] = cinn_sycl_tan_fp32(x[((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2))]);
        };
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
