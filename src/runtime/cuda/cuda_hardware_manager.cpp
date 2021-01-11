/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/runtime/cuda/cuda_hardware_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <cuda_runtime_api.h>
#include <exception>
#include <cstdlib>

namespace hipsycl {
namespace rt {

cuda_hardware_manager::cuda_hardware_manager(hardware_platform hw_platform)
    : _hw_platform(hw_platform) {

  int num_devices = 0;

  auto err = cudaGetDeviceCount(&num_devices);
  if (err != cudaSuccess) {
    num_devices = 0;

    if(err != cudaErrorNoDevice) {
      print_warning(
          __hipsycl_here(),
          error_info{"cuda_hardware_manager: Could not obtain number of devices",
                    error_code{"CUDA", err}});
    }
  }
  
  for (int dev = 0; dev < num_devices; ++dev) {
    _devices.push_back(std::move(cuda_hardware_context{dev}));
  }

}


std::size_t cuda_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context *cuda_hardware_manager::get_device(std::size_t index) {
  if (index >= _devices.size()){
    register_error(__hipsycl_here(),
                   error_info{"cuda_hardware_manager: Attempt to access invalid "
                              "device detected."});
    return nullptr;
  }

  return &(_devices[index]);
}

device_id cuda_hardware_manager::get_device_id(std::size_t index) const {
  if (index >= _devices.size()){
    register_error(__hipsycl_here(),
                   error_info{"cuda_hardware_manager: Attempt to access invalid "
                              "device detected."});
  }

  return device_id{backend_descriptor{_hw_platform, api_platform::cuda},
                   static_cast<int>(index)};
}


cuda_hardware_context::cuda_hardware_context(int dev) : _dev{dev} {
  _properties = std::make_unique<cudaDeviceProp>();
  auto err = cudaGetDeviceProperties(_properties.get(), dev);

  if (err != cudaSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"cuda_hardware_manager: Could not query device properties ",
                   error_code{"CUDA", err}});
  }
}

bool cuda_hardware_context::is_cpu() const {
  return !is_gpu();
}

bool cuda_hardware_context::is_gpu() const {
  return true;
}

std::size_t cuda_hardware_context::get_max_kernel_concurrency() const {
  return _properties->concurrentKernels + 1;
}

std::size_t cuda_hardware_context::get_max_memcpy_concurrency() const {
  // TODO: Modern CUDA as asyncEngineCount
  return get_max_kernel_concurrency();
}

std::string cuda_hardware_context::get_device_name() const {
  return _properties->name;
}

std::string cuda_hardware_context::get_vendor_name() const {
  return "NVIDIA";
}

bool cuda_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return false;
    break;
  case device_support_aspect::host_unified_memory:
    return false;
    break;
  case device_support_aspect::error_correction:
    return false; // TODO
    break;
  case device_support_aspect::global_mem_cache:
    return true;
    break;
  case device_support_aspect::global_mem_cache_read_only:
    return false;
    break;
  case device_support_aspect::global_mem_cache_write_only:
    return false;
    break;
  case device_support_aspect::images:
    return false;
    break;
  case device_support_aspect::little_endian:
    return true;
    break;
  }
  assert(false && "Unknown device aspect");
  std::terminate();
}

std::size_t
cuda_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return _properties->multiProcessorCount;
    break;
  case device_uint_property::max_global_size0:
    return _properties->maxThreadsPerBlock * _properties->maxGridSize[0];
    break;
  case device_uint_property::max_global_size1:
    return _properties->maxThreadsPerBlock * _properties->maxGridSize[1];
    break;
  case device_uint_property::max_global_size2:
    return _properties->maxThreadsPerBlock * _properties->maxGridSize[2];
    break;
  case device_uint_property::max_group_size:
    return _properties->maxThreadsPerBlock;
    break;
  case device_uint_property::preferred_vector_width_char:
    return 4;
    break;
  case device_uint_property::preferred_vector_width_double:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_float:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_half:
    return 2;
    break;
  case device_uint_property::preferred_vector_width_int:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_long:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_short:
    return 2;
    break;
  case device_uint_property::native_vector_width_char:
    return 4;
    break;
  case device_uint_property::native_vector_width_double:
    return 1;
    break;
  case device_uint_property::native_vector_width_float:
    return 1;
    break;
  case device_uint_property::native_vector_width_half:
    return 2;
    break;
  case device_uint_property::native_vector_width_int:
    return 1;
    break;
  case device_uint_property::native_vector_width_long:
    return 1;
    break;
  case device_uint_property::native_vector_width_short:
    return 2;
    break;
  case device_uint_property::max_clock_speed:
    return _properties->clockRate / 1000;
    break;
  case device_uint_property::max_malloc_size:
    return _properties->totalGlobalMem;
    break;
  case device_uint_property::address_bits:
    return 64;
    break;
  case device_uint_property::max_read_image_args:
    return 0;
    break;
  case device_uint_property::max_write_image_args:
    return 0;
    break;
  case device_uint_property::image2d_max_width:
    return 0;
    break;
  case device_uint_property::image2d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_width:
    return 0;
    break;
  case device_uint_property::image3d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_depth:
    return 0;
    break;
  case device_uint_property::image_max_buffer_size:
    return 0;
    break;
  case device_uint_property::image_max_array_size:
    return 0;
    break;
  case device_uint_property::max_samplers:
    return 0;
    break;
  case device_uint_property::max_parameter_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::mem_base_addr_align:
    return 8; // TODO
    break;
  case device_uint_property::global_mem_cache_line_size:
    return 128; //TODO
    break;
  case device_uint_property::global_mem_cache_size:
    return _properties->l2CacheSize; // TODO
    break;
  case device_uint_property::global_mem_size:
    return _properties->totalGlobalMem;
    break;
  case device_uint_property::max_constant_buffer_size:
    return _properties->totalConstMem;
    break;
  case device_uint_property::max_constant_args:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::local_mem_size:
    return _properties->sharedMemPerBlock;
    break;
  case device_uint_property::printf_buffer_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::partition_max_sub_devices:
    return 0;
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::string cuda_hardware_context::get_driver_version() const {
  int driver_version = 0;

  auto err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"cuda_hardware_manager: Querying driver version failed",
                   error_code{"CUDA", err}});
  }
  
  return std::to_string(driver_version);
}

std::string cuda_hardware_context::get_profile() const {
  return "FULL_PROFILE";
}

cuda_hardware_context::~cuda_hardware_context(){}


}
}