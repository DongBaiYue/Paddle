#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
#include <glog/logging.h>
namespace cinn {
namespace runtime {
namespace Sycl {
SYCLBackendAPI* SYCLBackendAPI::Global() {
  static auto* inst = new SYCLBackendAPI();
  return inst;
}

Target::Arch SYCLBackendAPI::Init(Target::Arch arch) {
  if (initialized_) return this->arch;
  // Target::Arch -> sycl::backend
  sycl::backend backend;
  switch (arch) {
    case Target::Arch::Unk:
      SYCL_CALL(backend =
                    sycl::device::get_devices(sycl::info::device_type::gpu)[0]
                        .get_backend());
      break;
    case Target::Arch::NVGPU:
      backend = sycl::backend::ext_oneapi_cuda;
      break;
    case Target::Arch::AMDGPU:
      backend = sycl::backend::ext_oneapi_hip;
      break;
    case Target::Arch::IntelGPU:
      backend = sycl::backend::ext_oneapi_level_zero;
      break;
    default:
      LOG(FATAL) << "SYCL Not supported arch:" << arch;
  }
  // look for matched devices
  for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (device.get_backend() == backend) {
      this->devices.push_back(device);
    }
  }
  if (this->devices.size() == 0) {
    LOG(FATAL) << "No valid gpu device matched given arch:" << arch;
  }
  this->contexts.resize(this->devices.size(), nullptr);
  this->queues.resize(this->devices.size());
  // sycl::backend -> Target::Arch
  switch (backend) {
    case sycl::backend::ext_oneapi_cuda:
      this->arch = Target::Arch::NVGPU;
      break;
    case sycl::backend::ext_oneapi_hip:
      this->arch = Target::Arch::AMDGPU;
      break;
    case sycl::backend::ext_oneapi_level_zero:
      this->arch = Target::Arch::IntelGPU;
      break;
    default:
      LOG(FATAL) << "SYCL Not supported arch:" << arch;
  }
  initialized_ = true;
  return this->arch;
}

void SYCLBackendAPI::set_device(int device_id) {
  if (!initialized_) Init(Target::Arch::Unk);
  if (device_id < 0) {
    LOG(FATAL) << "set valid device id! device id:" << device_id;
  } else if (device_id > this->devices.size() - 1) {
    LOG(FATAL) << "set valid device id! device id:" << device_id
               << " > max device id:" << this->devices.size() - 1;
  }
  if (this->contexts[device_id] == nullptr) {
    auto exception_handler = [](sycl::exception_list exceptions) {
      for (const std::exception_ptr& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const sycl::exception& e) {
          std::cout << "Caught asynchronous SYCL exception:\n"
                    << e.what() << std::endl;
        }
      }
    };
    sycl::property_list q_prop{
        sycl::property::queue::in_order()};  // In order queue
    // create context and queue
    this->contexts[device_id] =
        new sycl::context(this->devices[device_id], exception_handler);
    // one device one queue
    this->queues[device_id].push_back(new sycl::queue(
        *this->contexts[device_id], this->devices[device_id], q_prop));
  }
  this->now_device_id = device_id;
}

void* SYCLBackendAPI::malloc(size_t numBytes) {
  if (now_device_id == -1) set_device(0);
  VLOG(3) << "sycl malloc";
  void* dev_mem = nullptr;
  SYCL_CALL(dev_mem = sycl::malloc_device(
      numBytes, this->devices[now_device_id], *this->contexts[now_device_id]));
  if (dev_mem == nullptr)
    LOG(ERROR) << "allocate sycl device memory failure!" << std::endl;
  return dev_mem;
}

void SYCLBackendAPI::free(void* data) {
  VLOG(3) << "sycl free";
  SYCL_CALL(sycl::free(data, *this->contexts[now_device_id]));
}

void SYCLBackendAPI::memset(void* data, int value, size_t numBytes) {
  VLOG(3) << "sycl memset";
  SYCL_CALL(
      this->queues[now_device_id][0]->memset(data, value, numBytes).wait());
}

void SYCLBackendAPI::memcpy(void* dest,
                            const void* src,
                            size_t numBytes,
                            MemcpyType type) {
  VLOG(3) << "sycl memcpy";
  sycl::queue* Q;
  switch (type) {
    case MemcpyType::HostToHost:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::HostToDevice:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::DeviceToHost:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::DeviceToDevice:
      Q = this->queues[now_device_id][0];
      break;
  }
  SYCL_CALL(Q->memcpy(dest, src, numBytes).wait());
}

void SYCLBackendAPI::device_sync() {
  VLOG(3) << "sycl device sync";
  if (now_device_id == -1) set_device(0);
  for (auto queues_in_one_device : this->queues) {
    for (auto queue : queues_in_one_device) {
      SYCL_CALL(queue->wait_and_throw());
    }
  }
}

sycl::queue* SYCLBackendAPI::get_now_queue() {
  return this->queues[now_device_id][0];
}

std::string SYCLBackendAPI::GetGpuVersion() {
  if (now_device_id == -1) set_device(0);
  sycl::device device = this->devices[now_device_id];
  sycl::backend backend = device.get_backend();
  switch (backend) {
    case sycl::backend::ext_oneapi_cuda: {
      std::string gpu_version = "sm_";
      std::string version_with_point =
          device.get_info<sycl::info::device::backend_version>();
      size_t pos = version_with_point.find(".");
      if (pos != std::string::npos) {
        gpu_version +=
            version_with_point.substr(0, pos) +
            version_with_point.substr(pos + 1, version_with_point.size());
      }
      return gpu_version;
    }
    case sycl::backend::ext_oneapi_hip: {
      std::string gpu_version = device.get_info<sycl::info::device::version>();
      size_t pos = gpu_version.find(":");
      if (pos != std::string::npos) gpu_version = gpu_version.substr(0, pos);
      return gpu_version;
    }
    case sycl::backend::ext_oneapi_level_zero:
      return "";
    default:
      LOG(ERROR) << "unknown sycl backend!";
  }
}
}  // namespace Sycl
}  // namespace runtime
}  // namespace cinn