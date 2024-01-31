// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/cinn/backends/cuda_util.h"

namespace cinn {
namespace runtime {
namespace cuda {
class CUDABackendAPI final : public BackendAPI {
 public:
  CUDABackendAPI(){};
  ~CUDABackendAPI(){};
  static CUDABackendAPI* Global();
  void set_device(int device_id) final;
  // void set_active_devices(std::vector<int> device_ids) final;
  int get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id = std::nullopt) final;
  void* malloc(size_t numBytes) final;
  void free(void* data) final;
  void memset(void* data, int value, size_t numBytes) final;
  void memcpy(void* dest, const void* src, size_t numBytes, MemcpyType type) final;
  void device_sync() final;
 private:
  // now_device_id, change by set_device()
  int now_device_id = -1;
};
}  // namespace cuda
}  // namespace runtime
}  // namespace cinn