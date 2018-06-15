#ifndef LLVM_TOOLS_LLVM_MC_PREISELTRAN_H
#define LLVM_TOOLS_LLVM_MC_PREISELTRAN_H

#include "llvm/Pass.h"

namespace llvm {
  FunctionPass *createMCTuningPass();
} // namespace llvm

#endif
