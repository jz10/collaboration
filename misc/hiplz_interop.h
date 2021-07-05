#ifndef __RUNTEST_H__
#define __RUNTEST_H__

extern "C" {
  // Initialize HipLZ via providing native runtime information
  int hiplzInit(void* driverPtr, void* deviePtr, void* contextPtr, void* queueptr);

  // Run GEMM test via HipLZ
  int hipMatrixMultiplicationTest();
}

#endif
