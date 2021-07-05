#include "level_zero/ze_api.h"
// #include "CL/sycl/detail/pi.hpp"
#include "CL/sycl/backend/level_zero.hpp"
#include "oneapi/mkl.hpp"
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <stdio.h>
#include "ze_utils.h"

using namespace std;

int test_mkl(sycl::queue& my_queue) {
  vector<double> A(1000, 1);
  vector<double> B(1000, 2);
  vector<double> C(1000, 0);
  int m, n, k;
  m = n = k = 10;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = 10;
  double alpha = 1.0;
  double beta  = 1.1;
  
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

int test_onemkl(unsigned long hDriver, unsigned long hDevice, unsigned long hContext) {
  sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>((ze_driver_handle_t)hDriver);
 
  // make devices from converted platform and L0 device 
  sycl::device sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform,
								  (ze_device_handle_t) hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context = sycl::level_zero::make<sycl::context>(devices,
								     (ze_context_handle_t)hContext);
  sycl::queue queue = sycl::queue(sycl_context, sycl_device);

  // Test the oneMKL  
  return test_mkl(queue);
}

int main(int argc, char* argv[]) {
  ze_result_t my_errno;          
  printf(">>> Initializing L0 Platform and Device...\n");
  // Select the first GPU avalaible  
  // Initialize the driver  
  my_errno  = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  check_error(my_errno, "zeInit");
  // Discover all the driver instances 
  uint32_t driverCount = 0;
  my_errno = zeDriverGet(&driverCount, NULL);
  check_error(my_errno, "zeDriverGet");
  //Now where the phDrivers 
  ze_driver_handle_t* phDrivers = (ze_driver_handle_t*) malloc(driverCount * sizeof(ze_driver_handle_t));
  my_errno = zeDriverGet(&driverCount, phDrivers);
  check_error(my_errno, "zeDriverGet");
  // Device who will be selecter
  ze_device_handle_t hDevice = NULL;
  ze_driver_handle_t hDriver = NULL;
  for(uint32_t driver_idx = 0; driver_idx < driverCount; driver_idx++) {
    hDriver = phDrivers[driver_idx];
    /* - - - - 
    Device 
    - - - - */
    // if count is zero, then the driver will update the value with the total number of devices available.   
    uint32_t deviceCount = 0;
    my_errno = zeDeviceGet(hDriver, &deviceCount, NULL);
    check_error(my_errno, "zeDeviceGet");
    ze_device_handle_t* phDevices = (ze_device_handle_t*) malloc(deviceCount * sizeof(ze_device_handle_t));
    my_errno = zeDeviceGet(hDriver, &deviceCount, phDevices);
    check_error(my_errno, "zeDeviceGet");
    for(uint32_t device_idx = 0;  device_idx < deviceCount; device_idx++) {
      ze_device_properties_t device_properties;
      my_errno = zeDeviceGetProperties(phDevices[device_idx], &device_properties);
      check_error(my_errno, "zeDeviceGetProperties");
      if (device_properties.type == ZE_DEVICE_TYPE_GPU){
	printf("Running on Device #%d %s who is a GPU. \n", device_idx, device_properties.name);
	hDevice = phDevices[device_idx];
	break;
      }
    }
    free(phDevices);
    if (hDevice != NULL) {
        break;
    }
  }
  free(phDrivers);
  
  ze_context_handle_t hContext = NULL;
  // Create context   
  ze_context_desc_t context_desc = {
     ZE_STRUCTURE_TYPE_CONTEXT_DESC,
     NULL,
     0};
  my_errno = zeContextCreate(hDriver, &context_desc, &hContext);
  check_error(my_errno, "zeContextCreate");

  test_onemkl((unsigned long)hDriver, (unsigned long)hDevice, (unsigned long)hContext);
  
  // strt converting to SYCL  
  // make a sycl platform from L0 driver
  sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>(hDriver);
  // sycl::platform sycl_platform = sycl::level_zero::make_platform((pi_native_handle)hDriver); // sycl::make_platform<sycl::backend::level_zero>(hDriver);
  // make devices from converted platform and L0 device   
  sycl::device sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context = sycl::level_zero::make<sycl::context>(devices, hContext);
  sycl::queue queue{sycl_context, sycl_device};

  // Test the oneMKL
  test_mkl(queue);
  
  return 0;
}
