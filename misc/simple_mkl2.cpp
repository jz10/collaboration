// Standard SYCL header
#include <CL/sycl.hpp>
// STL classes
#include <exception>
#include <iostream>
// Declarations for Intel oneAPI Math Kernel Library DPC++ APIs
#include "oneapi/mkl.hpp"

#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  //
  // User obtains data here for A, B, C matrices, along with setting m, n, k, ldA, ldB, ldC.
  //
  // For this example, A, B and C should be initially stored in a std::vector,
  //   or a similar container having data() and size() member functions.
  //
  vector<double> A(1000, 1);
  vector<double> B(1000, 2);
  vector<double> C(1000, 0);
  int m, n, k;
  m = n = k = 10;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = 10;
  double alpha = 1.0;
  double beta  = 1.1;
  
  // Create GPU device 
  sycl::device my_device;
  try {
    my_device = sycl::device(sycl::gpu_selector());
  }
  catch (...) {
    std::cout << "Warning: GPU device not found! Using default device instead." << std::endl;
  }


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
    
    oneapi::mkl::blas::column_major::asum(my_queue, 10, A_buffer, 99, C_buffer);
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
  

  //
  // post process results
  //
  // Access data from C buffer and print out part of C matrix
  auto C_accessor = C_buffer.template get_access<sycl::access::mode::read>();
  std::cout << "\t --- " << C[0] << " = [ " << C_accessor[0] << ", "
	    << C_accessor[1] << ", ... ]\n";
  std::cout << "\t    [ " << C_accessor[1 * ldC + 0] << ", "
	    << C_accessor[1 * ldC + 1] << ",  ... ]\n";
  std::cout << "\t    [ " << "... ]\n";
  std::cout << std::endl;
  
  return 0;
}
