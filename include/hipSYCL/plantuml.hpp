#ifndef PLANTUML_HPP
#define PLANTUML_HPP

#include <iostream>
#include <vector>

#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/operations.hpp"

namespace hipsycl {
namespace rt {

class plantuml {
 private:
  std::vector<std::pair<std::size_t, std::size_t>> dependencies;
  int call = 0;

  void printOperationType(dag_node_ptr node) {

    if(node->get_operation()->is_requirement()) {
      std::cout << "REQUIREMENT" << std::endl;
    }
    if (typeid(*node->get_operation()) == typeid(kernel_operation)) {
      kernel_operation* k = dynamic_cast<kernel_operation*>(node->get_operation());

      std::cout << "\tkernel ";

      switch (k->get_launcher().find_launcher(node->get_assigned_device().get_backend())->get_kernel_type()) {
        case kernel_type::single_task:
          std::cout << "\tsingle_task";
          break;
        case kernel_type::basic_parallel_for:
          std::cout << "\tbasic_parallel_for";
          break;
        case kernel_type::ndrange_parallel_for:
          std::cout << "\tndrange_parallel_for";
          break;
        case kernel_type::hierarchical_parallel_for:
          std::cout << "\thierarchical_parallel_for";
          break;
        case kernel_type::scoped_parallel_for:
          std::cout << "\tscoped_parallel_for";
          break;
        case kernel_type::custom:
          std::cout << "\tcustom";
          break;
      }
    } else if (typeid(*node->get_operation()) == typeid(memset_operation)) {
      std::cout << "\tmemset";
    } else if (typeid(*node->get_operation()) == typeid(prefetch_operation)) {
      std::cout << "\tprefetch";
    } else if (typeid(*node->get_operation()) == typeid(memcpy_operation)) {
      std::cout << "\tmemcpy";
    } else if (typeid(*node->get_operation()) == typeid(buffer_memory_requirement)) {
      buffer_memory_requirement* b = dynamic_cast<buffer_memory_requirement*>(node->get_operation());
      std::cout << "\tbuffer memory";
      std::cout << "\n\t\taccess mode: " << b->get_access_mode();
      std::cout << "\n\t\telement size: " << b->get_element_size();
      auto range = b->get_data_region()->get_num_elements();
      std::cout << "\n\t\tnum elements: " << range[0] << "x" << range[1] << "x" << range[2];
      std::cout << "\n\t\tacces target: " << b->get_access_target();
      std::cout << "\n\t\thas device data location: " << b->has_device_data_location();
    }
  }

  void printBackend(dag_node_ptr node) {
    if (node->get_assigned_device().is_host()) {
      std::cout << "\tHOST";
    }

    switch (node->get_assigned_device().get_backend()) {
      case backend_id::omp:
        std::cout << "\tOMP";
        break;
      case backend_id::cuda:
        std::cout << "\tCUDA";
        break;
      case backend_id::hip:
        std::cout << "\tHIP";
        break;
    }
  }

  void printNode(dag_node_ptr node) {
    std::cout << "class N" << node->get_node_id() << " {" << std::endl;
    printOperationType(node);
    std::cout << std::endl;
    printBackend(node);

    std::cout << std::endl << "}" << std::endl;
  }

  void printReq(dag_node_ptr node, dag_node_ptr req) {
    if (req->get_operation()->is_data_transfer()) {
      std::cout << "N" << node->get_node_id() << "<-- \"data_transfer\""
                << "N" << req->get_node_id() << std::endl;
    } else if (req->get_operation()->is_requirement()) {
      std::cout << "N" << node->get_node_id() << "<-- \"req\""
                << "N" << req->get_node_id() << std::endl;
    } else {
      std::cout << "N" << node->get_node_id() << "<--"
                << "N" << req->get_node_id() << std::endl;
    }
  }

  void addNode(dag_node_ptr node) {
    static int count = 0;
    if (!node->has_node_id()) {
      node->assign_node_id(count);
      printNode(node);
      count++;
    }
  }

  void addDependency(dag_node_ptr node, dag_node_ptr req) {
    auto dep = std::make_pair(node->get_node_id(), req->get_node_id());
    if (!std::count(dependencies.begin(), dependencies.end(), dep)) {
      dependencies.push_back(dep);
      std::cout << "N" << dep.first << " <-- "
                << "N" << dep.second << " :" << call << std::endl;
    }
  }

  void recReq(dag_node_ptr node, dag_node_ptr req) {
    addNode(req);
    addDependency(node, req);

    for (auto req2 : req->get_requirements()) {
      recReq(req, req2);
    }
  }

 public:
  void print(dag dag) {
    std::cout << "package  \"dag_" << call << "\" #DDDDDD {" << std::endl;
    if (true) {  // Recursive dependencies
      for (auto node : dag.get_command_groups()) {
        addNode(node);
        std::cout << "commandgroupdag" << call << " -> N" << node->get_node_id() << std::endl; 
        for (auto req : node->get_requirements()) {
          recReq(node, req);
        }
      }
    } else {  // Just first dependency
      for (auto node : dag.get_command_groups()) {
        addNode(node);
        for (auto req : node->get_requirements()) {
          addNode(req);
          addDependency(node, req);
        }
      }
    }
    std::cout << std::endl << "}" << std::endl;

    call++;
  }
};

}
}

#endif