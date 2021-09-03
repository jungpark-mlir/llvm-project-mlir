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
#include "mlir/Dialect/GPU/GPUDialect.h"
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
    auto resultType = results[0].getType().template cast<MemRefType>();

    rewriter.setInsertionPoint(op);
    auto resultAlloc = rewriter.create<AllocOp>(loc, resultType);
    
    auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
    SmallVector<Value, 8> operands(op.getOperands());
    operands.push_back(resultAlloc);

    //auto fusedFuncOp = op->getCallee();
    // Try to find the referenced function.
    SmallVector<IntegerAttr, 5> globalSizeAttr;
    SmallVector<IntegerAttr, 5> localSizeAttr;

    auto fusedFuncOp =
      op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
    fusedFuncOp.walk([&](Operation *Lop) {
      llvm::errs()<< "visiting op : " << Lop->getName().getStringRef() << "\n";
      if (!isa<gpu::LaunchOp>(Lop))
        return;
      //               Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,               Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
      globalSizeAttr.push_back(Lop->getAttr("gridSizeZ");
      globalSizeAttr.push_back(Lop->getAttr("gridSizeY");
      globalSizeAttr.push_back(Lop->getAttr("gridSizeX");
      localSizeAttr.push_back(Lop->getAttr("blockSizeZ");
      localSizeAttr.push_back(Lop->getAttr("blockSizeY");
      localSizeAttr.push_back(Lop->getAttr("blockSizeX");
    });

    auto cop = rewriter.create<mlir::migraphx::CodeObjOp>(loc, resultType, operands);
    cop->setAttr("kernel", fnAttr);

/*
    globalSizeAttr.push_back(rewriter.getI64IntegerAttr(4));
    globalSizeAttr.push_back(rewriter.getI64IntegerAttr(4));
    globalSizeAttr.push_back(rewriter.getI64IntegerAttr(128));

    localSizeAttr.push_back(rewriter.getI64IntegerAttr(1));
    localSizeAttr.push_back(rewriter.getI64IntegerAttr(1));
    localSizeAttr.push_back(rewriter.getI64IntegerAttr(32));
*/
    cop->setAttr("globalSize",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(globalSizeAttr.begin(), globalSizeAttr.end())));
    cop->setAttr("localSize",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(localSizeAttr.begin(), localSizeAttr.end())));

    rewriter.replaceOp(op, cop->getResults());

    return success();
  }
};

} // namespace

void mlir::migraphx::populateFuncToCOBJPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<FuncToCOBJPattern>(context);
}
