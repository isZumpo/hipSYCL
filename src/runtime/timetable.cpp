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
  // Add kernel if not in timetable
  if (_table.count(kernel_name) <= 0) {
    // Tuple is of <count, sum, average>
    _table[kernel_name] = {{device, std::make_tuple(1, time, time)}};
    return;
  }

  auto device_table = _table[kernel_name];
  // If kernel entry already exists for device, update count, sum and average
  if (device_table.count(device) > 0) {

    auto tuple = device_table[device];
    ++std::get<0>(tuple);
    std::get<1>(tuple) += time;
    auto average = std::get<2>(tuple);
    std::get<2>(tuple) = (time + average) / 2.f;

  } else {
    device_table[device] = std::make_tuple(1, time, time);
  }
  _table[kernel_name] = device_table;
}

int timetable::get_count(std::string kernel_name, device_id device) {
  if (_table.count(kernel_name) > 0 && _table[kernel_name].count(device) > 0) {
    return std::get<0>(_table[kernel_name][device]);
  }

  return -1.0;
}

float timetable::get_sum(std::string kernel_name, device_id device) {
  if (_table.count(kernel_name) > 0 && _table[kernel_name].count(device) > 0) {
    return std::get<1>(_table[kernel_name][device]);
  }

  return -1.0;
}

float timetable::get_average(std::string kernel_name, device_id device) {
  if (_table.count(kernel_name) > 0 && _table[kernel_name].count(device) > 0) {
    return std::get<2>(_table[kernel_name][device]);
  }

  return -1.0;
}


}
}
