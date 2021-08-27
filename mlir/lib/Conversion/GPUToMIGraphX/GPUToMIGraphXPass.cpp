//===- TosaToMIGraphXPass.cpp - Lowering Tosa to MIGraphX Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToMIGraphX/TosaToMIGraphX.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace {

struct FuncToCOBJ
    : public FuncToCOBJPatternBase<FuncToCOBJPattern> {
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

std::unique_ptr<Pass> mlir::tosa::createTosaToMIGraphXRandom() {
  return std::make_unique<TosaToMIGraphXRandom>();
}
void mlir::tosa::addTosaToMIGraphXRandomPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createTosaToMIGraphXRandom());
}
