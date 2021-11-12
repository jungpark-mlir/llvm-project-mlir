//===- AlignTiling.cpp - Align Linalg ops with MIOpen ops
//------------------===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This pass refactors linalg.generic ops from global scope to tiled scope
// based on miopen lowering step2.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/FunctionSupport.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct MainWrapperPass
    : public MainWrapperPassBase<MainWrapperPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void MainWrapperPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp module = getOperation();
  auto ops = module.getOps<FuncOp>();
  OpBuilder b(ctx);

  for (auto f : module.getOps<FuncOp>()) {
    Location loc = f.getLoc();
    b.setInsertionPoint(f);
    auto type = f.getType();

    auto mainFunc = b.create<FuncOp>(loc, "main", type);
    b.setInsertionPointToStart(mainFunc.addEntryBlock());
    CallOp callOp = b.create<CallOp>(loc, f, mainFunc.getArguments());
    b.create<ReturnOp>(loc, callOp->getValue());
    //mlir::function_like_impl::eraseFunctionResults(mainFunc, {0}, 1, mainFunc.getTypeWithoutArgsAndResults({}, {0}));
  }
}

std::unique_ptr<Pass> mlir::miopen::createMainWrapperPass() {
  return std::make_unique<MainWrapperPass>();
}
