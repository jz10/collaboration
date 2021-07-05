// Standard SYCL header
#include <CL/sycl.hpp>
// STL classes
#include <exception>
#include <iostream>
// Declarations for Intel oneAPI Math Kernel Library DPC++ APIs
#include "oneapi/mkl.hpp"

#include <vector>

using namespace std;

int sycl_gemm(std::vector<float>& A, std::vector<vector>& B, std::vector<float>& C,
	      int m, int n, int k, double alpha, double beta) {
  int ldA, ldB, ldC;
  ldA = ldB = ldC = m;

  auto my_exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
	std::rethrow_exception(e);
      }
      catch (sycl::exception const& e) {
	std::cout << "Caught asynchronous SYCL exception:\n"
	<< e.what() << std::endl;
      }
      catch (std::exception const& e) {
	std::cout << "Caught asynchronous STL exception:\n"
	<< e.what() << std::endl;
      }
    }
  };
  
  // create execution queue on my gpu device with exception handler attached
  sycl::queue my_queue; // (my_device, my_exception_handler);
  // create sycl buffers of matrix data for offloading between device and host
  sycl::buffer<double, 1> A_buffer(A.data(), A.size());
  sycl::buffer<double, 1> B_buffer(B.data(), B.size());
  sycl::buffer<double, 1> C_buffer(C.data(), C.size());
  // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous exceptions
  try {
    oneapi::mkl::blas::column_major::gemm(my_queue,
					  oneapi::mkl::transpose::nontrans,
					  oneapi::mkl::transpose::nontrans,
					  m, n, k, alpha,
					  A_buffer, ldA, B_buffer, ldB, beta, C_buffer,
					  ldC);
  }
  catch (sycl::exception const& e) {
  std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
	      << e.what() << std::endl;
  }
  catch (std::exception const& e) {
    std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
	      << e.what() << std::endl;
  }
  // ensure any asynchronous exceptions caught are handled before proceeding
  my_queue.wait_and_throw();
  
  return 0;
}
