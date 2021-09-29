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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"

using namespace mlir;

namespace {
  /*
class FuncToCOBJPattern : public OpRewritePattern<CallOp> {
public:
  using OpRewritePattern<CallOp>::OpRewritePattern;
*/
class FuncToCOBJPattern : public OpConversionPattern<CallOp> {
//class FuncToCOBJPattern : public ConvertOpToLLVMPattern<CallOpType>{}
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto resultType = results[0].getType().template cast<MemRefType>();

    // Insert alloc for result buffer
    rewriter.setInsertionPoint(op);
    auto resultAlloc = rewriter.create<memref::AllocOp>(loc, resultType);
    
    // 
    auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
    SmallVector<Value, 8> mrOperands(op.getOperands());
    SmallVector<Value, 8> cobjArgs;
    mrOperands.push_back(resultAlloc);

    SmallVector<IntegerAttr, 5> globalSizeAttr;
    SmallVector<IntegerAttr, 5> localSizeAttr;
    SymbolRefAttr kernelRefAttr;

    auto fusedFuncOp =
      op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
      //op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(fnAttr.getValue());
    fusedFuncOp.walk([&](Operation *Wop) {
      if (!isa<gpu::LaunchFuncOp>(Wop)) {
        //llvm::errs()<< "visiting op : " << Wop->getName().getStringRef() << "\n";
        return;
      }

      auto Lop = cast<gpu::LaunchFuncOp>(Wop);
      // x, y, z
      auto gridSize = Lop.getGridSizeOperandValues();
      auto blockSize = Lop.getBlockSizeOperandValues();

      // FIXME: Better way to import attribute as I64?
      globalSizeAttr.push_back(rewriter.getI64IntegerAttr((((gridSize.z.getDefiningOp())->getAttrOfType<IntegerAttr>("value"))).getInt()));
      globalSizeAttr.push_back(rewriter.getI64IntegerAttr((((gridSize.y.getDefiningOp())->getAttrOfType<IntegerAttr>("value"))).getInt()));
      globalSizeAttr.push_back(rewriter.getI64IntegerAttr((((gridSize.x.getDefiningOp())->getAttrOfType<IntegerAttr>("value"))).getInt()));
      localSizeAttr.push_back(rewriter.getI64IntegerAttr((((blockSize.z.getDefiningOp())->getAttrOfType<IntegerAttr>("value"))).getInt()));
      localSizeAttr.push_back(rewriter.getI64IntegerAttr((((blockSize.y.getDefiningOp())->getAttrOfType<IntegerAttr>("value"))).getInt()));
      localSizeAttr.push_back(rewriter.getI64IntegerAttr((((blockSize.x.getDefiningOp())->getAttrOfType<IntegerAttr>("value"))).getInt()));

      kernelRefAttr = Lop->getAttrOfType<SymbolRefAttr>("kernel");

      // Lowering memref structure
      for (auto arg: mrOperands) {
        // Sending the reference to the memref itself because we're sending this to an excution engine which will handle the allocation.
        // allocation ptr
        cobjArgs.push_back(arg);
        // aligned ptr
        cobjArgs.push_back(arg);
        ValueRange noArgs({});
        
        // offset
        auto offsetOp = rewriter.create<mlir::migraphx::ConstantOp>(loc, rewriter.getI64Type(), noArgs)
        offsetOp->setAttr("value", rewriter.getI64IntegerAttr(0));
        cobjArgs.push_back(offsetOp);

        // shape
        auto argType = arg.getType();
        auto argShape = argType.getShape()

        for (auto dim: argShape) {
          auto constOp = rewriter.create<mlir::migraphx::ConstantOp>(loc, rewriter.getI64Type(), noArgs)
          constOp->setAttr("value", rewriter.getI64IntegerAttr(dim));
          cobjArgs.push_back(constOp);
        }

        // stride
        uint64_t stride = 1;
        for (auto dim: argShape) {
          auto constOp = rewriter.create<mlir::migraphx::ConstantOp>(loc, rewriter.getI64Type(), noArgs)
          constOp->setAttr("value", rewriter.getI64IntegerAttr(stride));
          cobjArgs.push_back(constOp);
          stride *= dim;
        }
      }
    });

    auto cop = rewriter.create<mlir::migraphx::CodeObjOp>(loc, resultType, cobjArgs);
    cop->setAttr("kernel", kernelRefAttr);    
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
