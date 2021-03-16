/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include "hipSYCL/runtime/dag_hybrid_scheduler.hpp"

#include <algorithm>
#include <iterator>
#include <vector>

#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/performance_model.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"

namespace hipsycl {
namespace rt {

namespace {
void abort_submission(dag_node_ptr node) {
  for (auto req : node->get_requirements()) {
    if (!req->is_submitted()) {
      req->cancel();
    }
  }
  node->cancel();
}

template <class Handler>
void execute_if_buffer_requirement(dag_node_ptr node, Handler h) {
  if (node->get_operation()->is_requirement()) {
    if (cast<requirement>(node->get_operation())->is_memory_requirement()) {
      if (cast<memory_requirement>(node->get_operation())->is_buffer_requirement()) {
        h(cast<buffer_memory_requirement>(node->get_operation()));
      }
    }
  }
}

// Initialize memory accesses for requirements
void initialize_memory_access(buffer_memory_requirement *bmem_req, device_id target_dev) {
  assert(bmem_req);

  void *device_pointer = bmem_req->get_data_region()->get_memory(target_dev);
  bmem_req->initialize_device_data(device_pointer);
  HIPSYCL_DEBUG_INFO << "dag_hybrid_scheduler: Preparing deferred pointer of "
                        "requirement node "
                     << dump(bmem_req) << std::endl;
}

result ensure_allocation_exists(buffer_memory_requirement *bmem_req, device_id target_dev) {
  assert(bmem_req);
  if (!bmem_req->get_data_region()->has_allocation(target_dev)) {
    const std::size_t num_bytes =
        bmem_req->get_data_region()->get_num_elements().size() * bmem_req->get_data_region()->get_element_size();
    const std::size_t min_align =  // max requested alignment the size of a sycl::vec<double, 16>
        std::min(bmem_req->get_data_region()->get_element_size(), sizeof(double) * 16);

    void *ptr = application::get_backend(target_dev.get_backend()).get_allocator(target_dev)->allocate(min_align, num_bytes);

    if (!ptr)
      return register_error(__hipsycl_here(), error_info{"dag_hybrid_scheduler: Lazy memory allocation has failed.",
                                                         error_type::memory_allocation_error});

    bmem_req->get_data_region()->add_empty_allocation(target_dev, ptr);
  }

  return make_success();
}

void for_each_explicit_operation(dag_node_ptr node, std::function<void(operation *)> explicit_op_handler) {
  if (node->is_submitted()) return;

  if (!node->get_operation()->is_requirement()) {
    explicit_op_handler(node->get_operation());
    return;
  } else {
    execute_if_buffer_requirement(node, [&](buffer_memory_requirement *bmem_req) {
      device_id target_device = node->get_assigned_device();

      std::vector<range_store::rect> outdated_regions;
      bmem_req->get_data_region()->get_outdated_regions(target_device, bmem_req->get_access_offset3d(),
                                                        bmem_req->get_access_range3d(), outdated_regions);

      for (range_store::rect region : outdated_regions) {
        std::vector<std::pair<device_id, range_store::rect>> update_sources;

        bmem_req->get_data_region()->get_update_source_candidates(target_device, region, update_sources);

        if (update_sources.empty()) {
          register_error(__hipsycl_here(), error_info{"dag_hybrid_scheduler: Could not obtain data "
                                                      "update sources when trying to materialize "
                                                      "implicit requirement"});
          node->cancel();
          return;
        }

        // Just use first source for now:
        memory_location src{update_sources[0].first, update_sources[0].second.first, bmem_req->get_data_region()};
        memory_location dest{target_device, region.first, bmem_req->get_data_region()};
        memcpy_operation op{src, dest, region.second};

        explicit_op_handler(&op);
      }
    });
  }
}

backend_executor *select_executor(dag_node_ptr node, operation *op) {
  device_id dev = node->get_assigned_device();

  assert(!op->is_requirement());
  backend_id executor_backend;
  device_id preferred_device;
  if (op->has_preferred_backend(executor_backend, preferred_device))
    // If we want an executor from a different backend, we may need to pass
    // a different device id.
    return application::get_backend(executor_backend).get_executor(preferred_device);
  else {
    return application::get_backend(dev.get_backend()).get_executor(dev);
  }
}

void submit(backend_executor *executor, dag_node_ptr node, operation *op) {
  std::vector<dag_node_ptr> reqs;
  node->for_each_nonvirtual_requirement([&](dag_node_ptr req) { reqs.push_back(req); });
  // Compress requirements by removing double entries and complete requirements
  reqs.erase(std::remove_if(reqs.begin(), reqs.end(), [](dag_node_ptr elem) { return elem->is_complete(); }), reqs.end());
  std::sort(reqs.begin(), reqs.end());
  reqs.erase(std::unique(reqs.begin(), reqs.end()), reqs.end());
  // TODO we can even eliminate more requirements, e.g.
  // node -> A -> B
  // node -> B
  // the dependency on B can be eliminated because it is already covered by A.
  // TODO: This might be better implemented in the dag_builder
  node->assign_to_executor(executor);

  executor->submit_directly(node, op, reqs);
}

result submit_requirement(dag_node_ptr req) {
  if (!req->get_operation()->is_requirement() || req->is_submitted()) return make_success();

  sycl::access::mode access_mode = sycl::access::mode::read_write;

  // Make sure that all required allocations exist
  // (they must exist when we try initialize device pointers!)
  result res = make_success();
  execute_if_buffer_requirement(req, [&](buffer_memory_requirement *bmem_req) {
    res = ensure_allocation_exists(bmem_req, req->get_assigned_device());
    access_mode = bmem_req->get_access_mode();
  });
  if (!res.is_success()) return res;

  // Then initialize memory accesses
  execute_if_buffer_requirement(
      req, [&](buffer_memory_requirement *bmem_req) { initialize_memory_access(bmem_req, req->get_assigned_device()); });

  // Don't create memcopies if access is discard
  if (access_mode != sycl::access::mode::discard_write && access_mode != sycl::access::mode::discard_read_write) {
    bool has_initialized_content = true;
    execute_if_buffer_requirement(req, [&](buffer_memory_requirement *bmem_req) {
      has_initialized_content =
          bmem_req->get_data_region()->has_initialized_content(bmem_req->get_access_offset3d(), bmem_req->get_access_range3d());
    });
    if (has_initialized_content) {
      for_each_explicit_operation(req, [&](operation *op) {
        if (!op->is_data_transfer()) {
          res = make_error(__hipsycl_here(), error_info{"dag_hybrid_scheduler: only data transfers are supported "
                                                        "as operations generated from implicit requirements.",
                                                        error_type::feature_not_supported});
        } else {
          backend_executor *executor = select_executor(req, op);
          // TODO What if we need to copy between two device backends through
          // host?
          submit(executor, req, op);
        }
      });
    } else {
      HIPSYCL_DEBUG_WARNING << "dag_hybrid_scheduler: Detected a requirement that is neither of "
                               "discard access mode (SYCL 1.2.1) nor noinit property (SYCL 2020) "
                               "that accesses uninitialized data. Consider changing to "
                               "discard/noinit. Optimizing potential data transfers away."
                            << std::endl;
    }
  }
  if (!res.is_success()) return res;

  // If the requirement did not result in any operations...
  if (!req->get_event()) {
    // create dummy event
    req->mark_virtually_submitted();
  }
  // This must be executed even if the requirement did
  // not result in actual operations in order to make sure
  // that regions are valid after discard accesses
  execute_if_buffer_requirement(req, [&](buffer_memory_requirement *bmem_req) {
    if (access_mode == sycl::access::mode::read) {
      bmem_req->get_data_region()->mark_range_valid(req->get_assigned_device(), bmem_req->get_access_offset3d(),
                                                    bmem_req->get_access_range3d());
    } else {
      bmem_req->get_data_region()->mark_range_current(req->get_assigned_device(), bmem_req->get_access_offset3d(),
                                                      bmem_req->get_access_range3d());
    }
  });

  return make_success();
}
}

// Lists and saves all currently avaliable backend devices.
void dag_hybrid_scheduler::initialize_devices() {
  application::get_runtime().backends().for_each_backend([this](backend *b) {
    std::size_t num_devices = b->get_hardware_manager()->get_num_devices();
    for (std::size_t dev = 0; dev < num_devices; ++dev) {
      this->_devices.push_back(b->get_hardware_manager()->get_device_id(dev));
    }
  });

  if (this->_devices.empty()) {
    register_error(__hipsycl_here(), error_info{"dag_hybrid_scheduler: No devices available", error_type::runtime_error});
  }
}

void dag_hybrid_scheduler::submit(dag dag) {
  device_id target_device;
  if (_devices.empty()) {
    initialize_devices();
    std::cout << "Found " << _devices.size() << " devices" << std::endl;
  }

  // static direct_model model();
  static estimate_execution_model model(_devices);

  model.assign_devices(dag);

  for (auto node : dag.get_command_groups()) {
    for (auto req : node->get_requirements()) {
      if (!req->get_operation()->is_requirement()) {
        if (!req->is_submitted()) {
          register_error(__hipsycl_here(), error_info{"dag_hybrid_scheduler: Hybrid scheduler does not "
                                                      "support processing multiple unsubmitted nodes",
                                                      error_type::feature_not_supported});
          abort_submission(node);
          return;
        }
      } else {
        result res = submit_requirement(req);

        if (!res.is_success()) {
          register_error(res);
          abort_submission(node);
          return;
        }
      }
    }

    if (node->get_operation()->is_requirement()) {
      result res = submit_requirement(node);

      if (!res.is_success()) {
        register_error(res);
        abort_submission(node);
        return;
      }
    } else {
      // hipSYCL: TODO What if this is an explicit copy between two device backends through
      // host?
      backend_executor *exec = select_executor(node, node->get_operation());
      rt::submit(exec, node, node->get_operation());
    }
    // Register node as submitted with the runtime
    // (only relevant for queue::wait() operations)
    application::dag().register_submitted_ops(node);
  }
}
}
}
