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

    // Insert alloc for result buffer
    rewriter.setInsertionPoint(op);
    auto resultAlloc = rewriter.create<memref::AllocOp>(loc, resultType);
    
    // 
    auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
    SmallVector<Value, 8> operands(op.getOperands());
    SmallVector<Value, 8> kernelArgs();
    operands.push_back(resultAlloc);

    SmallVector<IntegerAttr, 5> globalSizeAttr;
    SmallVector<IntegerAttr, 5> localSizeAttr;
    SymbolRefAttr kernelRefAttr;

    auto fusedFuncOp =
      op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
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

      auto Lloc = Lop.getLoc();
      auto numKernelOperands = Lop.getNumKernelOperands();
      auto arguments = getTypeConverter()->promoteOperands(
          Lloc, Lop.getOperands().take_back(numKernelOperands),
          operands.take_back(numKernelOperands), builder);
      auto numArguments = arguments.size();
      SmallVector<Type, 4> argumentTypes;
      argumentTypes.reserve(numArguments);
      for (auto argument : arguments)
        argumentTypes.push_back(argument.getType());
      for (auto en : llvm::enumerate(arguments)) {
        auto index = builder.create<LLVM::ConstantOp>(
            Lloc, llvmInt32Type, builder.getI32IntegerAttr(en.index()));
        kernelArgs.push_back(en.value());
      }

    });

    auto cop = rewriter.create<mlir::migraphx::CodeObjOp>(loc, resultType, kernelArgs);
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
