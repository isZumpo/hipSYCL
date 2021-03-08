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

#ifndef HIPSYCL_TIMETABLE_HPP
#define HIPSYCL_TIMETABLE_HPP

#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>

#include "hipSYCL/runtime/hardware.hpp"

namespace hipsycl {
namespace rt {

/**
 * @param count times the kernel has been executed.
 * @param sum the total time the kernel has been run.
 * @param average the average time the kernel takes to run.
 */
struct timetable_entry {
  int count; /**< times the kernel has been executed. */
  float sum; /**< the total time the kernel has been run. */
  float average; /**< the average time the kernel takes to run. */
};

class timetable {
 public:
  /**
   * Registers given time in timetable for device and kernel combination
   *
   * @param kernel_name the name of the kernel.
   * @param device which the kernel ran on.
   */
  void register_time(std::string kernel_name, device_id device, float time);

  timetable_entry get_entry(std::string kernel_name, device_id device);

  void print();

 private:
  std::map<std::string, std::unordered_map<device_id, timetable_entry>> _table;
  std::mutex _timetable_mutex;
};

}
}

#endif
