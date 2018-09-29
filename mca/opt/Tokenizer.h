//=- PassPrinters.h - Utilities to print analysis info for passes -*- C++ -*-=//
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

#ifndef LLVM_TOOLS_TOKENIZER_H
#define LLVM_TOOLS_TOKENIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;

FunctionPass *createFunctionTokenizer(const PassInfo *PI, raw_ostream &out,
				      bool Quiet);
} // end namespace llvm

#endif // LLVM_TOOLS_TOKENIZER_H
