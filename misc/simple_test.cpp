#include "level_zero/ze_api.h"
#include "CL/sycl/backend/level_zero.hpp"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
// #include "level_zero/ze_utils.h"
int main(int argc, char* argv[]) {
  ze_result_t my_errno;          
  printf(">>> Initializing OpenCL Platform and Device...\n");
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
  sycl::platform sycl_platform = sycl::make_platform<sycl::backend::level_zero>(hDriver);
  // make devices from converted platform and L0 device   
  sycl::device sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context = sycl::level_zero::make<sycl::context>(devices, hContext);
  sycl::queue queue{ sycl_context, sycl_device};
  // Create a command_group to issue command to the group. 
  // Use A lambda to generate the control group handler
  // Queue submision are asyncrhonous (similar to OpenMP nowait)                  
  queue.submit([&](sycl::handler &cgh) {
    // Create a output stream       
    sycl::stream sout(1024, 256, cgh);
    // Submit a unique task, using a lambda            
    cgh.single_task([=]() {
      sout << "Hello, World!" << sycl::endl;
    }); // End of the kernel function                 
  });   // End of the queue commands. The kernel is now submited                      
  // wait for all queue submissions to completion
  
  queue.wait();
  return 0;
}
