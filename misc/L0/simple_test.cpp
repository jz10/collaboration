#include "level_zero/ze_api.h"
#include "CL/sycl/backend/level_zero.hpp"
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <stdio.h>
#include "ze_utils.h"

class mod_image;

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
  // strt converting to SYCL  
  // make a sycl platform from L0 driver
  //  sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>(hDriver);
  sycl::platform sycl_platform = sycl::level_zero::make_platform(hDriver); // sycl::make_platform<sycl::backend::level_zero>(hDriver);
  // make devices from converted platform and L0 device   
  sycl::device sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context = sycl::level_zero::make<sycl::context>(devices, hContext);
  sycl::queue queue{sycl_context, sycl_device};
  // Create a command_group to issue command to the group. 
  // Use A lambda to generate the control group handler
  // Queue submision are asyncrhonous (similar to OpenMP nowait)                  
  /*queue.submit([&](sycl::handler &cgh) {
    // Create a output stream       
    sycl::stream sout(1024, 256, cgh);
    // Submit a unique task, using a lambda            
    cgh.single_task([=]() {
	sout << "Hello, World!" << sycl::endl;
    }); // End of the kernel function                 
    });*/

  auto array_size = 256;
  std::vector<float> A(array_size, 1.0f);
  std::vector<float> B(array_size, 1.0f);
  std::vector<float> C(array_size);
  auto A_buff = cl::sycl::buffer<float>(A.data(), cl::sycl::range<1>(array_size));
  auto B_buff = cl::sycl::buffer<float>(B.data(), cl::sycl::range<1>(array_size));
  auto C_buff = cl::sycl::buffer<float>(C.data(), cl::sycl::range<1>(array_size));
  auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
  auto work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
  auto total_threads = num_groups * work_group_size;
  queue.submit([&](cl::sycl::handler &cgh) {
      auto A_acc = A_buff.get_access<cl::sycl::access::mode::read>(cgh);
      auto B_acc = B_buff.get_access<cl::sycl::access::mode::read>(cgh);
      auto C_acc = C_buff.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for<class vec_add>(
				      cl::sycl::range<1>{total_threads}, [=](cl::sycl::item<1> itemId) {
					auto id = itemId.get_id(0);
					for (auto i = id; i < C_acc.get_count(); i += itemId.get_range()[0])
					  C_acc[i] = A_acc[i] + B_acc[i];
				      });
    });   // End of the queue commands. The kernel is now submited                      
// wait for all queue submissions to completion

  printf("ran test\n");
  
  queue.wait();
  return 0;
}
