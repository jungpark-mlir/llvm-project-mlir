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
    auto resultType = results[0].getType().template cast<MemRefType>();
  
    /*
    auto shape = resultType.getShape();
    SmallVector<IntegerAttr, 5> shapeAttr;
    for(auto dim: shape){
      shapeAttr.push_back(rewriter.getI32IntegerAttr(dim));
    }
*/
    rewriter.setInsertionPoint(op);
    auto resultAlloc = rewriter.create<AllocOp>(loc, resultType);
    
    auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
    SmallVector<Value, 8> operands(op.getOperands());
    operands.push_back(resultAlloc);

    //auto fusedFuncOp = op->getCallee();
    // Try to find the referenced function.
    auto fusedFuncOp =
         op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());

   // auto fusedRegion = fusedFuncOp->getRegions()[0];

    for (Region &region : op.getRegions()) {
      region.walk([&](Operation *Lop) {
        llvm::errs()<< "visiting op : " << Lop->getName().getStringRef() << "\n";
    });

    auto cop = rewriter.create<mlir::migraphx::CodeObjOp>(loc, resultType, operands);
    cop->setAttr("kernel", fnAttr);

    SmallVector<IntegerAttr, 5> globalSizeAttr;
    SmallVector<IntegerAttr, 5> localSizeAttr;

    globalSizeAttr.push_back(rewriter.getI64IntegerAttr(4));
    globalSizeAttr.push_back(rewriter.getI64IntegerAttr(4));
    globalSizeAttr.push_back(rewriter.getI64IntegerAttr(128));
    
    localSizeAttr.push_back(rewriter.getI64IntegerAttr(1));
    localSizeAttr.push_back(rewriter.getI64IntegerAttr(1));
    localSizeAttr.push_back(rewriter.getI64IntegerAttr(32));
    
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
