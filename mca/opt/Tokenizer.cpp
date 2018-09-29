//===- Tokenizer.cpp - Utilities to print analysis info for passes -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities to print analysis info for various kinds of passes.
///
//===----------------------------------------------------------------------===//

#include "Tokenizer.h"
#include "SlotTracker.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;
using namespace llvm;

namespace {

class Tokenizer {
public:
  Tokenizer(Function& F_) : F(F_) {
    InternalSlotTracker slotTracker(&F, true);
    // Setup slots for function values
    slotTracker.initializeIfNeeded();
    // Generate tokens for funciton values
    tokenize(slotTracker);
  }

  // Get the tokenized string
  string toString() {
    return funcName + " -- " + tokenized;
  }
  
protected:
  // Tokenize the function
  void tokenize(InternalSlotTracker& slotTracker) {
    // Get the demangled function name
    funcName = demangleFuncName(F, false);
    
    // Go through the function basic blocks
    ostringstream oss;
    // SlotTracker slotTracker = new SlotTracker(F);
    
    for (Function::iterator BI = F.begin(), E = F.end(); BI != E; ++ BI) {
      // Record basic block label
      BasicBlock* BB = &(* BI);
      oss << " BB" << slotTracker.getLocalSlot(BB) << " ";
      for (BasicBlock::iterator I = BI->begin(), E = BI->end(); I != E; ++I) {
	Instruction* Inst = &(* I);
	// Check instruction
	string instInfo = checkInst(Inst, slotTracker);
	if (instInfo != "") {
	  oss << " [ " << instInfo;
	  // Check the use values
	  checkOperands(Inst, oss, slotTracker);
	  oss << " ] ";
	}
      }
    }

    tokenized = oss.str();
  }

  // Check the instruction and generate the related tokens
  string checkInst(Instruction* I, InternalSlotTracker& slotTracker) {
    string res = "";

    // Check the uses
    Value::use_iterator UI = I->use_begin(), E = I->use_end();
    bool hasUses = (UI != E);
      
    unsigned opcode = I->getOpcode();
    if (Instruction::isCast(opcode))
      res = " cast ";
    else if(Instruction::isFuncletPad(opcode))
      res = " funcpad ";
    else if (Instruction::isBinaryOp(opcode))
      res = " bin ";
    else if (Instruction::isTerminator(opcode))
      res = " term ";
    else {
      switch (opcode) {
      case Instruction::PHI:
	res = " phi ";
	break;
      case Instruction::Load:
	res = " load ";
	break;
      case Instruction::Store:
	res = " store ";
	break;
      case Instruction::GetElementPtr:
	res = " gep ";
	break;
      case Instruction::Call:
      case Instruction::Invoke:
	res = " call ";
	break;
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
	res = " shift ";
	break;
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
	res = " lop ";
	break;
      case Instruction::Br:
      case Instruction::IndirectBr:
	res = " br ";
	break;
      case Instruction::CatchSwitch:
      case Instruction::CatchPad:
      case Instruction::CleanupPad:
      case Instruction::LandingPad:
      case Instruction::CatchRet:
      case Instruction::CleanupRet:
      case Instruction::Resume:
	res = " catch ";
	break;
      default:
	break;
      }
    }

    if (hasUses)
      return makeRetValue(I, slotTracker) + res;
    else
      return res;
  }

  // Check the operands for the given instruction and generate the related tokens
  void checkOperands(Instruction* I, ostringstream& oss, InternalSlotTracker& slotTracker) {
    for (unsigned i = 0; i < I->getNumOperands(); i ++) {
      Value* op = I->getOperand(i);
      oss << makeRetValue(op, slotTracker);
    }
  }

  // Make the return value name
  string makeRetValue(Value* V, InternalSlotTracker& slotTracker) {
    if (dyn_cast<Constant>(V))
      // We do not handle constant value so far 
      return "";
    else if (BasicBlock* BB = dyn_cast<BasicBlock>(V)){
      string BBName = "BB" + to_string((int)slotTracker.getLocalSlot(BB)) + " ";
      return BBName;
    } else if (Type* Ty = V->getType()) {
      if (Ty->isVoidTy()) {
	return "";
      } else {
	int slot = (int)slotTracker.getLocalSlot(V);
	string VName = getTypePrefix(V->getType(), slot) + " ";
	return VName;
      }
    } else
      return "";
  }

  // Get the type prefix
  string getTypePrefix(Type* Ty, int slot) {
    if (Ty->isIntegerTy())
      return "i" + to_string(slot);
    else if (Ty->isFloatingPointTy())
      return "d" + to_string(slot);
    else if (Ty->isLabelTy())
      return "l" + to_string(slot);
    else if (Ty->isFunctionTy())
      return "f" + to_string(slot);
    else if (Ty->isArrayTy())
      return "a" + to_string(slot);
    else if (Ty->isPointerTy())
      return "p" + to_string(slot);
    else if (Ty->isVectorTy())
      return "v" + to_string(slot);
    else if (Ty->isStructTy())
      return "s" + to_string(slot);
    
    return "";
  }
  
  // Demangle the function name
  string demangleFuncName(Function& F, bool need) {
    int status;
    string funcNameStr = F.getName();
    if (!need)
      return funcNameStr;
    
    char * realname = abi::__cxa_demangle(funcNameStr.c_str(), 0, 0, &status);
    if (realname) {
      string realName = realname;
      return realName;
    } else
      return F.getName();
  }
  
protected:
  // The function reference
  Function& F;

  // Function name
  string funcName;

  // Tokenized string
  string tokenized;
};
  
struct FunctionTokenizer : public FunctionPass {
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  static char ID;
  std::string PassName;
  bool QuietPass;

  FunctionTokenizer(const PassInfo *PI, raw_ostream &out, bool Quiet)
    : FunctionPass(ID), PassToPrint(PI), Out(out), QuietPass(Quiet) {
  }

  bool runOnFunction(Function &F) override {
    Tokenizer token(F);
    std::cout << token.toString() << std::endl;
    
    return false;
  }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char FunctionTokenizer::ID = 0;

} // end anonymous namespace

FunctionPass *llvm::createFunctionTokenizer(const PassInfo *PI,
					      raw_ostream &OS, bool Quiet) {
  // std::cout << "Create Tokenizer Pass" << std::endl;
  return new FunctionTokenizer(PI, OS, Quiet);
}
