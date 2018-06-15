set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  AsmPrinter
  CodeGen
  Core
  IRReader
  MC
  MIRParser
  ScalarOpts
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  )

# Support plugins.
set(LLVM_NO_DEAD_STRIP 1)

add_llvm_tool(llvm-mc-tuning
  llvm-mc-tuning.cpp
  PreISelTrans.cpp
  
  DEPENDS
  intrinsics_gen
  )
export_executable_symbols(llvm-mc-tuning)
