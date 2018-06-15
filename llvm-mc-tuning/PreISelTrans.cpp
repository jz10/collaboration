#include "PreISelTrans.h"

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  struct PreISelTuning : public FunctionPass {
    static char ID;
    PreISelTuning() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      errs() << "Hello: ";
      errs().write_escaped(F.getName()) << '\n'; 
      return false;
    }
  };
}

char PreISelTuning::ID = 0;
static RegisterPass<PreISelTuning> X("preisel_tuning", "PreISel Tuning Pass",
				     false /* Only looks at CFG */,
				     false /* Analysis Pass */);

FunctionPass * llvm::createMCTuningPass() {
  return new PreISelTuning();
}
