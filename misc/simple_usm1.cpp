#include <vector>

#include "level_zero/ze_api.h"
#include <CL/sycl.hpp>
#include "CL/sycl/backend/level_zero.hpp"

using namespace sycl;

const int size = 10;

void check_error(ze_result_t error, char const *name) {
  if (error != ZE_RESULT_SUCCESS) {
    fprintf(stderr, "Non-successful return code %d (%s) for %s.  Exiting.\n", error, "L0 API error", name);
    exit(EXIT_FAILURE);
  }
}

// Instantiate L0 related driver handler, device handler and context handler
int instantiateL0(ze_driver_handle_t& hDriver,
		  ze_device_handle_t& hDevice,
		  ze_context_handle_t& hContext) {
  ze_result_t my_errno;
  printf(">>> Initializing L0 Platform and Device...\n");
  // Select the first GPU avalaible
  // Initialize the driver  
  my_errno  = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  // Discover all the driver instan
  uint32_t driverCount = 0;
  my_errno = zeDriverGet(&driverCount, NULL);
  check_error(my_errno, "zeDriverGet");
  //Now where the phDrivers 
  ze_driver_handle_t* phDrivers = (ze_driver_handle_t*) malloc(driverCount * sizeof(ze_driver_handle_t));
  my_errno = zeDriverGet(&driverCount, phDrivers);
  check_error(my_errno, "zeDriverGet");
  // Device who will be selecter 
  // ze_device_handle_t hDevice = NULL;
  // ze_driver_handle_t hDriver = NULL;
  for(uint32_t driver_idx = 0; driver_idx < driverCount; driver_idx++) {
    hDriver = phDrivers[driver_idx]; 
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
  
  // ze_context_handle_t hContext = NULL;
  // Create context      
  ze_context_desc_t context_desc = {
     ZE_STRUCTURE_TYPE_CONTEXT_DESC,
     NULL,
     0};
  my_errno = zeContextCreate(hDriver, &context_desc, &hContext);
  check_error(my_errno, "zeContextCreate");
  
  return 0;
}

int main() {
  ze_driver_handle_t  hDriver  = NULL;
  ze_device_handle_t  hDevice  = NULL;
  ze_context_handle_t hContext = NULL;

  // Instantiate L0 handlers
  instantiateL0(hDriver, hDevice, hContext);

   // make a sycl platform from L0 driver
  sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>(hDriver);
  // make devices from converted platform and L0 device
  sycl::device sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context = sycl::level_zero::make<sycl::context>(devices, hContext);
  
  queue q{sycl_context, sycl_device};

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
