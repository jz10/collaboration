#include "SlotTracker.h"

using namespace llvm;

// Module level constructor. Causes the contents of the Module (sans functions)
// to be added to the slot table.
InternalSlotTracker::InternalSlotTracker(const Module *M, bool ShouldInitializeAllMetadata)
    : TheModule(M), ShouldInitializeAllMetadata(ShouldInitializeAllMetadata) {}

// Function level constructor. Causes the contents of the Module and the one
// function provided to be added to the slot table.
InternalSlotTracker::InternalSlotTracker(const Function *F, bool ShouldInitializeAllMetadata)
    : TheModule(F ? F->getParent() : nullptr), TheFunction(F),
      ShouldInitializeAllMetadata(ShouldInitializeAllMetadata) {}

InternalSlotTracker::InternalSlotTracker(const ModuleSummaryIndex *Index)
    : TheModule(nullptr), ShouldInitializeAllMetadata(false), TheIndex(Index) {}

void InternalSlotTracker::initializeIfNeeded() {
  if (TheModule) {
    processModule();
    TheModule = nullptr; ///< Prevent re-processing next time we're called.
  }

  if (TheFunction && !FunctionProcessed)
    processFunction();
}

void InternalSlotTracker::initializeIndexIfNeeded() {
  if (!TheIndex)
    return;
  processIndex();
  TheIndex = nullptr; ///< Prevent re-processing next time we're called.
}

// Iterate through all the global variables, functions, and global
// variable initializers and create slots for them.
void InternalSlotTracker::processModule() {
  // ST_DEBUG("begin processModule!\n");

  // Add all of the unnamed global variables to the value table.
  for (const GlobalVariable &Var : TheModule->globals()) {
    if (!Var.hasName())
      CreateModuleSlot(&Var);
    processGlobalObjectMetadata(Var);
    auto Attrs = Var.getAttributes();
    if (Attrs.hasAttributes())
      CreateAttributeSetSlot(Attrs);
  }

  for (const GlobalAlias &A : TheModule->aliases()) {
    if (!A.hasName())
      CreateModuleSlot(&A);
  }

  for (const GlobalIFunc &I : TheModule->ifuncs()) {
    if (!I.hasName())
      CreateModuleSlot(&I);
  }

  // Add metadata used by named metadata.
  for (const NamedMDNode &NMD : TheModule->named_metadata()) {
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      CreateMetadataSlot(NMD.getOperand(i));
  }

  for (const Function &F : *TheModule) {
    if (!F.hasName())
      // Add all the unnamed functions to the table.
      CreateModuleSlot(&F);

    if (ShouldInitializeAllMetadata)
      processFunctionMetadata(F);

    // Add all the function attributes to the table.
    // FIXME: Add attributes of other objects?
    AttributeSet FnAttrs = F.getAttributes().getFnAttributes();
    if (FnAttrs.hasAttributes())
      CreateAttributeSetSlot(FnAttrs);
  }

  // ST_DEBUG("end processModule!\n");
}

// Process the arguments, basic blocks, and instructions  of a function.
void InternalSlotTracker::processFunction() {
  // ST_DEBUG("begin processFunction!\n");
  fNext = 0;

  // Process function metadata if it wasn't hit at the module-level.
  if (!ShouldInitializeAllMetadata)
    processFunctionMetadata(*TheFunction);

  // Add all the function arguments with no names.
  for(Function::const_arg_iterator AI = TheFunction->arg_begin(),
      AE = TheFunction->arg_end(); AI != AE; ++AI)
    //xxx if (!AI->hasName())
    CreateFunctionSlot(&*AI);

  // ST_DEBUG("Inserting Instructions:\n");

  // Add all of the basic blocks and instructions with no names.
  for (auto &BB : *TheFunction) {
    //xxx if (!BB.hasName())
    CreateFunctionSlot(&BB);

    for (auto &I : BB) {
      //xxx if (!I.getType()->isVoidTy() && !I.hasName())
      CreateFunctionSlot(&I);

      // We allow direct calls to any llvm.foo function here, because the
      // target may not be linked into the optimizer.
      if (auto CS = ImmutableCallSite(&I)) {
        // Add all the call attributes to the table.
        AttributeSet Attrs = CS.getAttributes().getFnAttributes();
        if (Attrs.hasAttributes())
          CreateAttributeSetSlot(Attrs);
      }
    }
  }

  FunctionProcessed = true;

  // ST_DEBUG("end processFunction!\n");
}

// Iterate through all the GUID in the index and create slots for them.
void InternalSlotTracker::processIndex() {
  // ST_DEBUG("begin processIndex!\n");
  assert(TheIndex);

  // The first block of slots are just the module ids, which start at 0 and are
  // assigned consecutively. Since the StringMap iteration order isn't
  // guaranteed, use a std::map to order by module ID before assigning slots.
  std::map<uint64_t, StringRef> ModuleIdToPathMap;
  for (auto &ModPath : TheIndex->modulePaths())
    ModuleIdToPathMap[ModPath.second.first] = ModPath.first();
  for (auto &ModPair : ModuleIdToPathMap)
    CreateModulePathSlot(ModPair.second);

  // Start numbering the GUIDs after the module ids.
  GUIDNext = ModulePathNext;

  for (auto &GlobalList : *TheIndex)
    CreateGUIDSlot(GlobalList.first);

  for (auto &TId : TheIndex->typeIds())
    CreateGUIDSlot(GlobalValue::getGUID(TId.first));

  // ST_DEBUG("end processIndex!\n");
}

void InternalSlotTracker::processGlobalObjectMetadata(const GlobalObject &GO) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  GO.getAllMetadata(MDs);
  for (auto &MD : MDs)
    CreateMetadataSlot(MD.second);
}

void InternalSlotTracker::processFunctionMetadata(const Function &F) {
  processGlobalObjectMetadata(F);
  for (auto &BB : F) {
    for (auto &I : BB)
      processInstructionMetadata(I);
  }
}

void InternalSlotTracker::processInstructionMetadata(const Instruction &I) {
  // Process metadata used directly by intrinsics.
  if (const CallInst *CI = dyn_cast<CallInst>(&I))
    if (Function *F = CI->getCalledFunction())
      if (F->isIntrinsic())
        for (auto &Op : I.operands())
          if (auto *V = dyn_cast_or_null<MetadataAsValue>(Op))
            if (MDNode *N = dyn_cast<MDNode>(V->getMetadata()))
              CreateMetadataSlot(N);

  // Process metadata attached to this instruction.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  I.getAllMetadata(MDs);
  for (auto &MD : MDs)
    CreateMetadataSlot(MD.second);
}

/// Clean up after incorporating a function. This is the only way to get out of
/// the function incorporation state that affects get*Slot/Create*Slot. Function
/// incorporation state is indicated by TheFunction != 0.
void InternalSlotTracker::purgeFunction() {
  // ST_DEBUG("begin purgeFunction!\n");
  fMap.clear(); // Simply discard the function level map
  TheFunction = nullptr;
  FunctionProcessed = false;
  // ST_DEBUG("end purgeFunction!\n");
}

/// getGlobalSlot - Get the slot number of a global value.
int InternalSlotTracker::getGlobalSlot(const GlobalValue *V) {
  // Check for uninitialized state and do lazy initialization.
  initializeIfNeeded();

  // Find the value in the module map
  ValueMap::iterator MI = mMap.find(V);
  return MI == mMap.end() ? -1 : (int)MI->second;
}

/// getMetadataSlot - Get the slot number of a MDNode.
int InternalSlotTracker::getMetadataSlot(const MDNode *N) {
  // Check for uninitialized state and do lazy initialization.
  initializeIfNeeded();

  // Find the MDNode in the module map
  mdn_iterator MI = mdnMap.find(N);
  return MI == mdnMap.end() ? -1 : (int)MI->second;
}

/// getLocalSlot - Get the slot number for a value that is local to a function.
int InternalSlotTracker::getLocalSlot(const Value *V) {
  assert(!isa<Constant>(V) && "Can't get a constant or global slot with this!");

  // Check for uninitialized state and do lazy initialization.
  initializeIfNeeded();

  ValueMap::iterator FI = fMap.find(V);
  return FI == fMap.end() ? -1 : (int)FI->second;
}

int InternalSlotTracker::getAttributeGroupSlot(AttributeSet AS) {
  // Check for uninitialized state and do lazy initialization.
  initializeIfNeeded();

  // Find the AttributeSet in the module map.
  as_iterator AI = asMap.find(AS);
  return AI == asMap.end() ? -1 : (int)AI->second;
}

int InternalSlotTracker::getModulePathSlot(StringRef Path) {
  // Check for uninitialized state and do lazy initialization.
  initializeIndexIfNeeded();

  // Find the Module path in the map
  auto I = ModulePathMap.find(Path);
  return I == ModulePathMap.end() ? -1 : (int)I->second;
}

int InternalSlotTracker::getGUIDSlot(GlobalValue::GUID GUID) {
  // Check for uninitialized state and do lazy initialization.
  initializeIndexIfNeeded();

  // Find the GUID in the map
  guid_iterator I = GUIDMap.find(GUID);
  return I == GUIDMap.end() ? -1 : (int)I->second;
}

/// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
void InternalSlotTracker::CreateModuleSlot(const GlobalValue *V) {
  assert(V && "Can't insert a null Value into SlotTracker!");
  assert(!V->getType()->isVoidTy() && "Doesn't need a slot!");
  assert(!V->hasName() && "Doesn't need a slot!");

  unsigned DestSlot = mNext++;
  mMap[V] = DestSlot;

  // ST_DEBUG("  Inserting value [" << V->getType() << "] = " << V << " slot=" <<
  //         DestSlot << " [");
  // G = Global, F = Function, A = Alias, I = IFunc, o = other
  // ST_DEBUG((isa<GlobalVariable>(V) ? 'G' :
  //          (isa<Function>(V) ? 'F' :
  //           (isa<GlobalAlias>(V) ? 'A' :
  //            (isa<GlobalIFunc>(V) ? 'I' : 'o')))) << "]\n");
}

/// CreateSlot - Create a new slot for the specified value if it has no name.
void InternalSlotTracker::CreateFunctionSlot(const Value *V) {
  //xxx assert(!V->getType()->isVoidTy() && !V->hasName() && "Doesn't need a slot!");

  unsigned DestSlot = fNext++;
  fMap[V] = DestSlot;

  // G = Global, F = Function, o = other
  // ST_DEBUG("  Inserting value [" << V->getType() << "] = " << V << " slot=" <<
  //         DestSlot << " [o]\n");
}

/// CreateModuleSlot - Insert the specified MDNode* into the slot table.
void InternalSlotTracker::CreateMetadataSlot(const MDNode *N) {
  assert(N && "Can't insert a null Value into SlotTracker!");

  // Don't make slots for DIExpressions. We just print them inline everywhere.
  if (isa<DIExpression>(N))
    return;

  unsigned DestSlot = mdnNext;
  if (!mdnMap.insert(std::make_pair(N, DestSlot)).second)
    return;
  ++mdnNext;

  // Recursively add any MDNodes referenced by operands.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (const MDNode *Op = dyn_cast_or_null<MDNode>(N->getOperand(i)))
      CreateMetadataSlot(Op);
}

void InternalSlotTracker::CreateAttributeSetSlot(AttributeSet AS) {
  assert(AS.hasAttributes() && "Doesn't need a slot!");

  as_iterator I = asMap.find(AS);
  if (I != asMap.end())
    return;

  unsigned DestSlot = asNext++;
  asMap[AS] = DestSlot;
}

/// Create a new slot for the specified Module
void InternalSlotTracker::CreateModulePathSlot(StringRef Path) {
  ModulePathMap[Path] = ModulePathNext++;
}

/// Create a new slot for the specified GUID
void InternalSlotTracker::CreateGUIDSlot(GlobalValue::GUID GUID) {
  GUIDMap[GUID] = GUIDNext++;
}

