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

#include "hipSYCL/runtime/performance_model.hpp"

#include "hipSYCL/runtime/instrumentation.hpp"

namespace hipsycl {
namespace rt {

void random_model::assign_devices(dag &dag) {
  for (auto node : dag.get_command_groups()) {
    device_id target_device;
    if (node->get_operation()->is_requirement()) {
      if (!node->get_execution_hints().has_hint<hints::bind_to_device>()) {
        register_error(__hipsycl_here(), error_info{"random_model: Random performance model does not "
                                                    "support DAG nodes not bound to devices.",
                                                    error_type::feature_not_supported});
        abort_submission(node);
        return;
      }
      // Use execution hint device for buffer memory specific nodes.
      target_device = node->get_execution_hints().get_hint<hints::bind_to_device>()->get_device_id();
    } else {
      srand(rand() ^ time(NULL));
      target_device = _devices[rand() % _devices.size()];
    }
    node->assign_to_device(target_device);

    // Assign all requirments to the same kernel that requires them.
    for (auto req : node->get_requirements()) {
      if (req->is_complete() || req->is_submitted()) {
        // Seems to fix the issue of kernel and requirements are not assigned to the same device.
        continue;
      }
      req->assign_to_device(target_device);
    }
  }
}

void dynamic_model::assign_devices(dag &dag) {
  for (auto node : dag.get_command_groups()) {
    device_id target_device;
    if (node->get_operation()->is_requirement()) {
      if (!node->get_execution_hints().has_hint<hints::bind_to_device>()) {
        register_error(__hipsycl_here(), error_info{"dynamic_model: Dynamic performance model does not "
                                                    "support DAG nodes not bound to devices.",
                                                    error_type::feature_not_supported});
        abort_submission(node);
        return;
      }
      // Use execution hint device for buffer memory specific nodes.
      target_device = node->get_execution_hints().get_hint<hints::bind_to_device>()->get_device_id();
    } else {
      srand(rand() ^ time(NULL));
      target_device = _devices[rand() % _devices.size()];
    }
    node->assign_to_device(target_device);

    // Assign all requirments to the same kernel that requires them.
    for (auto req : node->get_requirements()) {
      if (req->is_complete() || req->is_submitted()) {
        // Seems to fix the issue of kernel and requirements are not assigned to the same device.
        continue;
      }
      req->assign_to_device(target_device);
    }

    // Listen to events and add to timetable
    if (!node->get_operation()->is_requirement()) {
      std::thread profilingThread(
          [](timetable *timetable, dag_node_ptr node) {
            node->get_operation()->get_instrumentations().instrument<rt::timestamp_profiler>();

            std::string kernel_name = "Unknown";
            if (typeid(*node->get_operation()) == typeid(kernel_operation)) {
              kernel_operation *k = dynamic_cast<kernel_operation *>(node->get_operation());
              kernel_name = k->_kernel_name;
            }

            auto &profiler = node->get_operation()->get_instrumentations().await<rt::timestamp_profiler>();
            auto start = profiler.await_event(rt::timestamp_profiler::event::operation_started);
            auto end = profiler.await_event(rt::timestamp_profiler::event::operation_finished);
            auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(end.time_since_epoch()).count() -
                         std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count());

            timetable->register_time(kernel_name, node->get_assigned_device(), duration);
          },
          _timetable.get(), node);

      profilingThread.detach();
    }
  }
}
}

}
