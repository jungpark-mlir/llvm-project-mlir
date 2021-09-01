//===- GPUToMIGraphX.cpp - Lowering GPU to MIGraphX Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the GPU to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToMIGraphX/GPUToMIGraphX.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class FuncToCOBJPattern : public OpRewritePattern<CallOp> {
public:
  using OpRewritePattern<CallOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(CallOp op,
                  PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto resultType = results[0].getType().template cast<ShapedType>();
    auto shape = resultType.getShape();
    SmallVector<IntegerAttr, 5> shapeAttr;
    for(auto dim: shape){
      shapeAttr.push_back(rewriter.getI32IntegerAttr(dim));
    }

    auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
    ValueRange args({});
    auto cop = rewriter.create<mlir::migraphx::CodeObjOp>(loc, resultType, args);
    cop->setAttr("kernel", fnAttr);
    cop->setAttr("globalSize", rewriter.getI64IntegerAttr(1024));
    cop->setAttr("localSize", rewriter.getI64IntegerAttr(128));
    
    rewriter.replaceOp(op, cop->getResults());

    return success();
  }
};

} // namespace

void mlir::migraphx::populateFuncToCOBJPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<FuncToCOBJPattern>(context);
}
