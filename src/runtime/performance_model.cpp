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
#include <algorithm>

#include "hipSYCL/runtime/performance_model.hpp"

#include "hipSYCL/runtime/instrumentation.hpp"

namespace hipsycl {
namespace rt {

void direct_model::assign_devices(dag &dag) {
  for (auto node : dag.get_command_groups()) {
    if (!node->get_execution_hints().has_hint<hints::bind_to_device>()) {
    register_error(__hipsycl_here(),
                   error_info{"dag_direct_scheduler: Direct scheduler does not "
                              "support DAG nodes not bound to devices.",
                              error_type::feature_not_supported});
    abort_submission(node);
    return;
   }

    device_id target_device = node->get_execution_hints()
                                  .get_hint<hints::bind_to_device>()
                                  ->get_device_id();
    node->assign_to_device(target_device);
    
    for (auto req : node->get_requirements()){
      if (!node->get_execution_hints().has_hint<hints::bind_to_device>()) {
      req->assign_to_device(target_device);
      } else {
      req->assign_to_device(node->get_execution_hints()
                                .get_hint<hints::bind_to_device>()
                                ->get_device_id());
      }
    }
  }
}

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

void estimate_execution_model::assign_devices(dag &dag) {
  std::unordered_map<device_id, float> dag_estimated_times;
  for (auto device : _devices) {
    dag_estimated_times.emplace(device, 0.0f);
  }

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
      std::string kernel_name = "Unknown";
      if (typeid(*node->get_operation()) == typeid(kernel_operation)) {
        kernel_operation *k = dynamic_cast<kernel_operation *>(node->get_operation());
        kernel_name = k->_kernel_name;
      }

      std::vector<device_id> missing_entries = _timetable->get_missing_entries(kernel_name);

      // Fill entries for kernel before using data for scheduling
      if (missing_entries.size() > 0) {
        target_device = missing_entries.front();
      } else {
        std::vector<std::pair<device_id, float>> sorted_average;
        float treshhold_value = 0.7f; // If the difference in execution time is within this threshhold
        for (auto device : _devices) {
          sorted_average.push_back({device,
                                          // _timetable->get_entry(kernel_name, device).average + dag_estimated_times[device]});
                                          _timetable->get_entry(kernel_name, device).average});
        }

        std::sort(sorted_average.begin(), sorted_average.end(),
                  [](const auto &l, const auto &r) { return l.second < r.second; });

        //Check if the execution time is within the threshhold or not
        float new_estimated_time;
        if(sorted_average[1].second * treshhold_value + dag_estimated_times[sorted_average[1].first] <
          sorted_average[0].second + dag_estimated_times[sorted_average[0].first]) {
          target_device = sorted_average[0].first; //If no dependecy is found, revert back to the fastest device.
          new_estimated_time = sorted_average[0].second + dag_estimated_times[target_device];
          for(auto req : node->get_requirements()){
            if(!req->get_requirements().empty()){
              target_device = req->get_requirements().front()->get_assigned_device();
              auto find = std::find_if(sorted_average.begin(), sorted_average.end(), [&target_device] (const auto &s) { return s.first == target_device; } );
              new_estimated_time = find->second + dag_estimated_times[find->first];          
              break;
            }
          }

        } else {
          target_device = sorted_average[0].first;
          new_estimated_time = sorted_average[0].second + dag_estimated_times[target_device];
        } 
        dag_estimated_times[target_device] = new_estimated_time;    
      }
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
