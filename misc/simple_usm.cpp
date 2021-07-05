#include <vector>

#include "level_zero/ze_api.h"
#include <CL/sycl.hpp>
#include "CL/sycl/backend/level_zero.hpp"

using namespace sycl;

const int size = 10;

int main() {
  queue q;

  // USM allocator for data of type int in shared memory
  typedef usm_allocator<float, usm::alloc::shared> vec_alloc;
  // Create allocator for device associated with q
  vec_alloc myAlloc(q);
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

  unsigned long total_threads = 128;
  
  q.submit([&](cl::sycl::handler &h) {
      h.parallel_for<class vec_add>(cl::sycl::range<1>{total_threads},
		     [=](id<1> idx) {
                       // auto idx = itemId.get_id(0);
		       C[idx] = A[idx] + B[idx];
		     });
    }).wait();

  for (int i = 0; i < size; i++) std::cout << c[i] << std::endl;
  return 0;
}
