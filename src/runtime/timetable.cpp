/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#include "hipSYCL/runtime/timetable.hpp"

namespace hipsycl {
namespace rt {

void timetable::register_time(std::string kernel_name, device_id device, float time) {
  std::lock_guard<std::mutex> lock(_timetable_mutex);

  // Add kernel if not in timetable
  if (_table.count(kernel_name) <= 0) {
    _table[kernel_name] = {{device, timetable_entry{.count = 1, .sum = time, .average = time}}};
    return;
  }

  auto device_table = _table[kernel_name];
  // If kernel entry already exists for device, update count, sum and average
  if (device_table.count(device) > 0) {
    auto entry = device_table[device];
    entry.count += 1;
    entry.sum += time;
    entry.average = entry.sum / entry.count;
    device_table[device] = entry;
  } else {
    device_table[device] = timetable_entry{.count = 1, .sum = time, .average = time};
  }
  _table[kernel_name] = device_table;
}

timetable_entry timetable::get_entry(std::string kernel_name, device_id device) {
  if (_table.count(kernel_name) > 0 && _table[kernel_name].count(device) > 0) {
    return _table[kernel_name][device];
  }

  return timetable_entry{.count = -1, .sum = -1.0, .average = -1.0};
}

std::vector<device_id> timetable::get_missing_entries(std::string kernel_name) {
  if (_table.count(kernel_name) <= 0) {
    return _devices;
  }

  if (_table[kernel_name].size() == _devices.size()) {
    return {};
  }

  std::vector<device_id> missing;
  for (auto &device : _devices) {
    if (_table[kernel_name].count(device) <= 0) {
      missing.push_back(device);
    }
  }

  return missing;
}

void timetable::print() {
  std::lock_guard<std::mutex> lock(_timetable_mutex);

  std::cout << "\nTimetable entries for all kernels: \n";
  float total_cuda_time = 0, total_omp_time = 0;
  for (const auto &kernel_device : _table) {
    std::cout << "-------------  " << kernel_device.first << std::endl;
    for (const auto &device_entry : kernel_device.second) {
      std::cout << "device id: " << device_entry.first.get_id() << " on backend ";
      switch (device_entry.first.get_backend()) {
        case backend_id::omp:
          std::cout << "OMP";
          total_omp_time += device_entry.second.sum;
          break;
        case backend_id::cuda:
          std::cout << "CUDA";
          total_cuda_time += device_entry.second.sum;
          break;
        case backend_id::hip:
          std::cout << "HIP";
          break;
        default:
          std::cout << "UNKNOWN";
          break;
      }

      std::cout << "\t[count: " << std::dec << device_entry.second.count << ", sum: " << device_entry.second.sum
                << ", average: " << device_entry.second.average << "]" << std::endl;
    }

    std::cout << "-------------" << std::endl;
  }

  std::cout << "Total cuda time: " << total_cuda_time << std::endl << "Total omp time: " << total_omp_time << std::endl;
  std::cout << "Total run time: " << total_omp_time + total_cuda_time << std::endl;
}
}
}
