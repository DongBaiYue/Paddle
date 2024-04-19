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
      // SYCL_CALL(backend =
      //               sycl::device::get_devices(sycl::info::device_type::gpu)[0].get_backend());
      backend = sycl::device::get_devices(sycl::info::device_type::gpu)[0].get_backend();                    
      break;
    case Target::Arch::NVGPU:
      backend = sycl::backend::cuda;
      break;
    case Target::Arch::AMDGPU:
      backend = sycl::backend::rocm;
      break;
    case Target::Arch::IntelGPU:
      backend = sycl::backend::level_zero;
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
    case sycl::backend::cuda:
      this->arch = Target::Arch::NVGPU;
      break;
    case sycl::backend::rocm:
      this->arch = Target::Arch::AMDGPU;
      break;
    case sycl::backend::level_zero:
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

int SYCLBackendAPI::get_device(){
  return this->now_device_id;
}

std::variant<int, std::array<int, 3>> SYCLBackendAPI::get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id) {
  int index = device_id.value_or(this->now_device_id);
  std::variant<int, std::array<int, 3>> rv;
  switch (device_property) {
    case DeviceProperty::MaxBlockDims: {
      // TODO fix it
      //sycl::id<3> max_work_item_sizes = this->devices[index].get_info<sycl::info::device::max_work_item_sizes<3>>();
      //rv = std::array<int, 3>{max_work_item_sizes[2], max_work_item_sizes[1], max_work_item_sizes[0]};
      rv = std::array<int, 3>{1024, 1024 , 1024};
      break;
    }
    case DeviceProperty::MaxGridDims: {
      //sycl::id<3> grid_dims = this->devices[index].get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>() / this->devices[index].get_info<sycl::info::device::max_work_item_sizes<3>>();
      rv = std::array<int, 3>{2097151, 1024, 1024};
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      rv = this->devices[index].get_info<sycl::info::device::local_mem_size>();
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      rv = this->devices[index].get_info<sycl::info::device::max_work_group_size>();
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      // LOG(FATAL) << "SYCL Not supported device property : MaxThreadsPerSM !";
      rv = this->devices[index].get_info<sycl::info::device::max_work_group_size>();
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      rv = this->devices[index].get_info<sycl::info::device::max_compute_units>();
      break;
    }
    case DeviceProperty:: MaxBlocksPerSM: {
      LOG(FATAL) << "SYCL Not supported device property : MaxBlocksPerSM !";
      break;
    }
    case DeviceProperty::WarpSize: {
      std::vector<size_t> sub_group_sizes = this->devices[index].get_info<sycl::info::device::sub_group_sizes>();
      size_t max_sub_group_size = *max_element(std::begin(sub_group_sizes), std::end(sub_group_sizes));
      rv = static_cast<int>(max_sub_group_size);
      break;
    }
    default:
      LOG(FATAL) << "Not supported device property!";
  }
  return rv;
}

void* SYCLBackendAPI::malloc(size_t numBytes) {
  if (now_device_id == -1) set_device(0);
  VLOG(3) << "sycl malloc";
  void* dev_mem = nullptr;
  // SYCL_CALL(dev_mem = sycl::malloc_device(
  //     numBytes, this->devices[now_device_id], *this->contexts[now_device_id]));
  dev_mem = sycl::malloc_device(
      numBytes, this->devices[now_device_id], *this->contexts[now_device_id]);
  if (dev_mem == nullptr)
    LOG(ERROR) << "allocate sycl device memory failure!" << std::endl;
  return dev_mem;
}

void SYCLBackendAPI::free(void* data) {
  VLOG(3) << "sycl free";
  // SYCL_CALL(sycl::free(data, *this->contexts[now_device_id]));
  sycl::free(data, *this->contexts[now_device_id]);
}

void SYCLBackendAPI::memset(void* data, int value, size_t numBytes) {
  VLOG(3) << "sycl memset";
  // SYCL_CALL(
  //     this->queues[now_device_id][0]->memset(data, value, numBytes).wait());
    this->queues[now_device_id][0]->memset(data, value, numBytes).wait();
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
  Q->memcpy(dest, src, numBytes).wait();
  // SYCL_CALL(Q->memcpy(dest, src, numBytes).wait());
}

void SYCLBackendAPI::device_sync() {
  VLOG(3) << "sycl device sync";
  if (now_device_id == -1) set_device(0);
  for (auto queues_in_one_device : this->queues) {
    for (auto queue : queues_in_one_device) {
      // SYCL_CALL(queue->wait_and_throw());
      queue->wait_and_throw();
    }
  }
}

void SYCLBackendAPI::stream_sync(void* stream){
  VLOG(3) << "sycl stream sync";
  // SYCL_CALL(static_cast<sycl::queue *>(stream)->wait_and_throw());
  static_cast<sycl::queue *>(stream)->wait_and_throw();
}

sycl::queue* SYCLBackendAPI::get_now_queue() {
  return this->queues[now_device_id][0];
}

sycl::context* SYCLBackendAPI::get_default_context() {
  return this->contexts[now_device_id];
}

sycl::device SYCLBackendAPI::get_default_device() {
  return this->devices[now_device_id];
}

std::string SYCLBackendAPI::GetGpuVersion() {
  if (now_device_id == -1) set_device(0);
  sycl::device device = this->devices[now_device_id];
  sycl::backend backend = device.get_backend();
  switch (backend) {
    case sycl::backend::cuda: {
      std::string gpu_version = "sm_";
      std::string version_with_point =
          device.get_info<sycl::info::device::version>();
          // device.get_info<sycl::info::device::backend_version>();
      size_t pos = version_with_point.find(".");
      if (pos != std::string::npos) {
        gpu_version +=
            version_with_point.substr(0, pos) +
            version_with_point.substr(pos + 1, version_with_point.size());
      }
      return gpu_version;
    }
    case sycl::backend::rocm: {
      std::string gpu_version = device.get_info<sycl::info::device::version>();
      size_t pos = gpu_version.find(":");
      if (pos != std::string::npos) gpu_version = gpu_version.substr(0, pos);
      return gpu_version;
    }
    // case sycl::backend::cnrt: {
    //   std::string gpu_version = "mtp_372";
    //   return gpu_version;
    // }
    case sycl::backend::level_zero:
      return "";
    default:
      LOG(ERROR) << "unknown sycl backend!";
  }
}
}  // namespace sycl
}  // namespace runtime
}  // namespace cinn