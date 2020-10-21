/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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


#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(accessor_tests, reset_device_fixture)


BOOST_AUTO_TEST_CASE(local_accessors) {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler& cgh) {
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 1, mode::read_write, target::local>
        {local_size, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
        cl::sycl::nd_range<1>{global_size, local_size},
        [=](cl::sycl::nd_item<1> item) {
          const auto lid = item.get_local_id(0);
          scratch[lid] = acc[item.get_global_id()];
          item.barrier();
          for(size_t i = local_size/2; i > 0; i /= 2) {
            if(lid < i) scratch[lid] += scratch[lid + i];
            item.barrier();
          }
          if(lid == 0) acc[item.get_global_id()] = scratch[0];
        });
    });
  }

  for(size_t i = 0; i < global_size / local_size; ++i) {
    size_t expected = 0;
    for(size_t j = 0; j < local_size; ++j) expected += i * local_size + j;
    size_t computed = host_buf[i * local_size];
    BOOST_TEST(computed == expected);
  }
}

BOOST_AUTO_TEST_CASE(placeholder_accessors) {
  using namespace cl::sycl::access;
  constexpr size_t num_elements = 4096 * 1024;

  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{num_elements};

  {
    auto acc = buf.get_access<mode::discard_write>();
    for(size_t i = 0; i < num_elements; ++i) acc[i] = static_cast<int>(i);
  }

  cl::sycl::accessor<int, 1, mode::read_write, target::global_buffer,
                     placeholder::true_t>
      ph_acc{buf};

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.require(ph_acc);
    cgh.parallel_for<class placeholder_accessors1>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        ph_acc[tid] *= 2;
      });
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    auto ph_acc_copy = ph_acc; // Test that placeholder accessors can be copied
    cgh.require(ph_acc_copy);
    cgh.parallel_for<class placeholder_accessors2>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        ph_acc_copy[tid] *= 2;
      });
  });

  {
    auto acc = buf.get_access<mode::read>();
    for(size_t i = 0; i < num_elements; ++i) {
      BOOST_REQUIRE(acc[i] == 4 * i);
    }
  }
}

// TODO: Extend this
BOOST_AUTO_TEST_CASE(accessor_api) {
  namespace s = cl::sycl;

  s::buffer<int, 1> buf_a(32);
  s::buffer<int, 1> buf_b(32);
  auto buf_c = buf_a;

  const auto run_test = [&](auto get_access) {
    auto acc_a1 = get_access(buf_a);
    auto acc_a2 = acc_a1;
    auto acc_a3 = get_access(buf_a, s::range<1>(16));
    auto acc_a4 = get_access(buf_a, s::range<1>(16), s::id<1>(4));
    auto acc_a5 = get_access(buf_a);
    auto acc_b1 = get_access(buf_b);
    auto acc_c1 = get_access(buf_c);

    BOOST_REQUIRE(acc_a1 == acc_a1);
    BOOST_REQUIRE(acc_a2 == acc_a2);
    BOOST_REQUIRE(acc_a1 != acc_a3);
    BOOST_REQUIRE(acc_a1 != acc_a4);
    BOOST_REQUIRE(acc_a1 != acc_b1);
    // NOTE: A strict reading of the 1.2.1 Rev 7 spec, section 4.3.2 would imply
    // that these should not be equal, as they are not copies of acc_a1.
    //
    // These tests are currently commented out because the expected results
    // differ between host and device accessors if the comparisons are
    // tested in command group scope instead of kernel scope.
    //BOOST_REQUIRE(acc_a1 == acc_a5);
    //BOOST_REQUIRE(acc_a1 == acc_c1);
  };

  // Test host accessors
  run_test([&](auto buf, auto... args) {
    return buf.template get_access<s::access::mode::read>(args...);
  });

  // Test device accessors
  s::queue queue;
  queue.submit([&](s::handler& cgh) {
    run_test([&](auto buf, auto... args) {
      return buf.template get_access<s::access::mode::read>(cgh, args...);
    });
    cgh.single_task<class accessor_api_device_accessors>([](){});
  });

  // Test local accessors
  queue.submit([&](s::handler& cgh) {
    s::accessor<int, 1, s::access::mode::read_write, s::access::target::local> acc_a(32, cgh);
    s::accessor<int, 1, s::access::mode::read_write, s::access::target::local> acc_b(32, cgh);
    auto acc_c = acc_a;

    BOOST_REQUIRE(acc_a == acc_a);
    BOOST_REQUIRE(acc_a != acc_b);
    BOOST_REQUIRE(acc_a == acc_c);

    cgh.parallel_for<class accessor_api_local_accessors>(s::nd_range<1>(1, 1), [](s::nd_item<1>){});
  });
}

BOOST_AUTO_TEST_CASE(nested_subscript) {
  namespace s = cl::sycl;
  s::queue q;
  
  s::range<2> buff_size2d{64,64};
  s::range<3> buff_size3d{buff_size2d[0],buff_size2d[1],64};
  
  s::buffer<int, 2> buff2{buff_size2d};
  s::buffer<int, 3> buff3{buff_size3d};
  
  q.submit([&](s::handler& cgh){
    auto acc = buff2.get_access<s::access::mode::discard_read_write>(cgh);
    
    cgh.parallel_for<class nested_subscript2d>(buff_size2d, [=](s::id<2> idx){
      size_t x = idx[0];
      size_t y = idx[1];
      // Writing
      acc[x][y] = static_cast<int>(x*buff_size2d[1] + y);
      // Reading and making sure access id the same as with operator[id<>]
      if(acc[x][y] != acc[idx])
        acc[x][y] = -1;
    });
  });
  
  q.submit([&](s::handler& cgh){
    auto acc = buff3.get_access<s::access::mode::discard_read_write>(cgh);
    
    cgh.parallel_for<class nested_subscript3d>(buff_size3d, [=](s::id<3> idx){
      size_t x = idx[0];
      size_t y = idx[1];
      size_t z = idx[2];
      // Writing
      acc[x][y][z] = static_cast<int>(x*buff_size3d[1]*buff_size3d[2] + y*buff_size3d[2] + z);
      // Reading and making sure access id the same as with operator[id<>]
      if(acc[x][y][z] != acc[idx])
        acc[x][y][z] = -1;
    });
  });
  
  auto host_acc2d = buff2.get_access<s::access::mode::read>();
  auto host_acc3d = buff3.get_access<s::access::mode::read>();
  
  for(size_t x = 0; x < buff_size3d[0]; ++x)
    for(size_t y = 0; y < buff_size3d[1]; ++y) {
       
      size_t linear_id2d = static_cast<int>(x*buff_size2d[1] + y);
      s::id<2> id2d{x,y};
      BOOST_CHECK(host_acc2d[id2d] == linear_id2d);
      BOOST_CHECK(host_acc2d.get_pointer()[linear_id2d] == linear_id2d);
        
      for(size_t z = 0; z < buff_size3d[2]; ++z) {
        size_t linear_id3d = x*buff_size3d[1]*buff_size3d[2] + y*buff_size3d[2] + z;
        s::id<3> id3d{x,y,z};
        BOOST_CHECK(host_acc3d[id3d] == linear_id3d);
        BOOST_CHECK(host_acc3d.get_pointer()[linear_id3d] == linear_id3d);
      }
    }
}

BOOST_AUTO_TEST_SUITE_END()