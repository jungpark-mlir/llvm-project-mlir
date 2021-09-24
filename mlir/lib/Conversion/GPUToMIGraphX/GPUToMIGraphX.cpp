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
    SmallVector<Value, 8> kernelArgs;
    SmallVector<Value, 8> cobjArgs;
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

      auto llvmFuncOp = 
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(kernelRefAttr);
      SmallVector<Value, 8> llvmArgs(llvmFuncOp.getOperands());

      auto Lloc = Lop.getLoc();
      for (auto argument : llvmArgs) {
        MemRefDescriptor desc(argument);
        kernelArgs.push_back(desc);
      }

      Type llvmInt32Type = IntegerType::get(getContext(), 32);

      for (auto en : llvm::enumerate(kernelArgs)) {
        auto index = rewriter.create<LLVM::ConstantOp>(
            Lloc, llvmInt32Type, rewriter.getI32IntegerAttr(en.index()));
        cobjArgs.push_back(en.value());
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
