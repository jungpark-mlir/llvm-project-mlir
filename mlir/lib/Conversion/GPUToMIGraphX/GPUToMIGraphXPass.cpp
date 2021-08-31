//===- GPUToMIGraphXPass.cpp - Lowering GPU to MIGraphX Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes GPU operations to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/GPUToMIGraphX/GPUToMIGraphX.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace {

struct GPUToMIGraphX
    : public GPUToMIGraphXPatternBase<GPUToMIGraphX> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<migraphx::MIGraphXDialect, StandardOpsDialect, gpu::GPUDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<gpu::GPUDialect, migraphx::MIGraphXDialect, StandardOpsDialect>();
    target.addIllegalOp<FuncOp>();
/*
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
*/
    FuncOp func = getFunction();
    mlir::migraphx::populateFuncToCOBJPatterns(
        func.getContext(), &patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::migraphx::createGPUToMIGraphX() {
  return std::make_unique<GPUToMIGraphX>();
}
void mlir::migraphx::addGPUToMIGraphXPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createGPUToMIGraphX());
}
