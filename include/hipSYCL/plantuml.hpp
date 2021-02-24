#ifndef PLANTUML_HPP
#define PLANTUML_HPP

#include <iostream>
#include <vector>
#include <fstream>

#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/operations.hpp"

namespace hipsycl {
namespace rt {

class plantuml {
 private:
  std::vector<std::pair<std::size_t, std::size_t>> dependencies;
  int call = 0;

  std::ofstream outputfile{"plantuml_output.plantuml"};

  void printOperationType (dag_node_ptr node) {
    if(node->get_operation()->is_requirement()) {
      outputfile << "R";
    } else {
      outputfile << "K";
    }
  }

  void printOperationInfo(dag_node_ptr node) {

    if(node->get_operation()->is_requirement()) {
      outputfile << "REQUIREMENT" << std::endl;
    }
    if (typeid(*node->get_operation()) == typeid(kernel_operation)) {
      kernel_operation* k = dynamic_cast<kernel_operation*>(node->get_operation());

      outputfile << "\tkernel ";

      switch (k->get_launcher().find_launcher(node->get_assigned_device().get_backend())->get_kernel_type()) {
        case kernel_type::single_task:
          outputfile << "\tsingle_task";
          break;
        case kernel_type::basic_parallel_for:
          outputfile << "\tbasic_parallel_for";
          break;
        case kernel_type::ndrange_parallel_for:
          outputfile << "\tndrange_parallel_for";
          break;
        case kernel_type::hierarchical_parallel_for:
          outputfile << "\thierarchical_parallel_for";
          break;
        case kernel_type::scoped_parallel_for:
          outputfile << "\tscoped_parallel_for";
          break;
        case kernel_type::custom:
          outputfile << "\tcustom";
          break;
      }
    } else if (typeid(*node->get_operation()) == typeid(memset_operation)) {
      outputfile << "\tmemset";
    } else if (typeid(*node->get_operation()) == typeid(prefetch_operation)) {
      outputfile << "\tprefetch";
    } else if (typeid(*node->get_operation()) == typeid(memcpy_operation)) {
      outputfile << "\tmemcpy";
    } else if (typeid(*node->get_operation()) == typeid(buffer_memory_requirement)) {
      buffer_memory_requirement* b = dynamic_cast<buffer_memory_requirement*>(node->get_operation());
      outputfile << "\tbuffer memory";
      outputfile << "\n\t\taccess mode: " << b->get_access_mode();
      outputfile << "\n\t\telement size: " << b->get_element_size();
      auto range = b->get_data_region()->get_num_elements();
      outputfile << "\n\t\tnum elements: " << range[0] << "x" << range[1] << "x" << range[2];
      outputfile << "\n\t\taccess target: " << b->get_access_target();
      outputfile << "\n\t\thas device data location: " << b->has_device_data_location();
    }
  }

  void printBackend(dag_node_ptr node) {
    if (node->get_assigned_device().is_host()) {
      outputfile << "\tHOST";
    }

    switch (node->get_assigned_device().get_backend()) {
      case backend_id::omp:
        outputfile << "\tOMP";
        break;
      case backend_id::cuda:
        outputfile << "\tCUDA";
        break;
      case backend_id::hip:
        outputfile << "\tHIP";
        break;
    }
  }

  void printBackendColor(dag_node_ptr node) {
    switch (node->get_assigned_device().get_backend()) {
      case backend_id::omp:
        outputfile << "#0e93cc";
        break;
      case backend_id::cuda:
        outputfile << "#14d111";
        break;
      case backend_id::hip:
        outputfile << "#cc0e21";
        break;
    }
  }

  void printNode(dag_node_ptr node) {
    outputfile << "class N" << node->get_node_id() << " << (";
    printOperationType(node);
    outputfile << ",";
    printBackendColor(node);
    outputfile << ") >> {\n";
    printOperationInfo(node);
    outputfile << std::endl;
    printBackend(node);

    outputfile << std::endl << "}" << std::endl;
  }

  void printReq(dag_node_ptr node, dag_node_ptr req) {
    if (req->get_operation()->is_data_transfer()) {
      outputfile << "N" << node->get_node_id() << "<-- \"data_transfer\""
                << "N" << req->get_node_id() << std::endl;
    } else if (req->get_operation()->is_requirement()) {
      outputfile << "N" << node->get_node_id() << "<-- \"req\""
                << "N" << req->get_node_id() << std::endl;
    } else {
      outputfile << "N" << node->get_node_id() << "<--"
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
      outputfile << "N" << dep.first << " <-- "
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


    outputfile << "package  \"dag_" << call << "\" #DDDDDD {" << std::endl;
    if (true) {  // Recursive dependencies
      for (auto node : dag.get_command_groups()) {
        addNode(node);
        // outputfile << "commandgroupdag" << call << " -> N" << node->get_node_id() << std::endl; 
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
    outputfile << std::endl << "}" << std::endl;

    call++;
  }
};

}
}

#endif