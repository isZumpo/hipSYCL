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

#ifndef HIPSYCL_PERFORMANCE_MODEL_HPP
#define HIPSYCL_PERFORMANCE_MODEL_HPP

#include <functional>

#include "dag.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/timetable.hpp"

namespace hipsycl {
namespace rt {

class performance_model {
 public:
  virtual ~performance_model() {}
  virtual void assign_devices(dag &dag){};

 protected:
  void abort_submission(dag_node_ptr node) {
    for (auto req : node->get_requirements()) {
      if (!req->is_submitted()) {
        req->cancel();
      }
    }
    node->cancel();
  }
};

class random_model : public performance_model {
 public:
  random_model(std::vector<device_id> &devices) : _devices(devices) {}
  void assign_devices(dag &dag);

 private:
  std::vector<device_id> _devices;
};

class dynamic_model : public performance_model {
 public:
  ~dynamic_model() {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    _timetable->print();
  }
  dynamic_model(std::vector<device_id> &devices) : _devices(devices) {
    _timetable = std::make_unique<timetable>(_devices);
  }
  void assign_devices(dag &dag);

 private:
  std::vector<device_id> _devices;
  std::unique_ptr<timetable> _timetable;
};

}
}

#endif
