#include <vector>
#include <iostream>

#include "level_zero/ze_api.h"
#include <CL/sycl.hpp>
#include "CL/sycl/backend/level_zero.hpp"

#include "sycl_hiplz_interop.h"

using namespace sycl;

const int WIDTH = 10;

const int size = 100;

// CPU implementation of matrix transpose
void matrixMultiplyCPUReference(const float * __restrict A,
                                const float * __restrict B,
                                float * __restrict C) {
  for (uint i = 0; i < WIDTH; i++) {
    for (uint j = 0; j < WIDTH; j++) {
      float acc = 0.0f;
      for (uint k = 0; k < WIDTH; k++) {
	acc += B[i*WIDTH + k] * A[k*WIDTH + j];
      }
      C[i*WIDTH + j] = acc;
    }
  }
}


int main() {
  queue myQueue;
  
  // USM allocator for data of type int in shared memory
  typedef usm_allocator<float, usm::alloc::shared> vec_alloc;
  // Create allocator for device associated with q
  vec_alloc myAlloc(myQueue);
  // Create std vectors with the allocator
  std::vector<float, vec_alloc >
    a(size, myAlloc),
    b(size,  myAlloc),
    c(size, myAlloc);

  // Get pointer to vector data for access in kernel
  auto A = a.data();
  auto B = b.data();
  auto C = c.data();
  
  for (int i = 0; i < size; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = i;
  }

 

  // Alloc memory on host
  std::vector<float> c_ref(size);
  auto C_ref = c_ref.data();
  // Get CPU result for refereence
  matrixMultiplyCPUReference(A, B, C_ref);

  int err = 0;
  for (int i = 0; i < size; i ++)
    err += (int)(c[i] - c_ref[i]) * 1000;

  if (err == 0) {
    std::cout << "FAIL: " << err << " failures \n";
  } else {
    std::cout << "PASSED\n";
  }

  return 0;
}
