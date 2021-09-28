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
    //SmallVector<Value, 8> operands(op.getOperands());
    SmallVector<Value, 8> llArgs;
    SmallVector<Value, 8> cobjArgs;
    //operands.push_back(resultAlloc);

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

      auto llvmFuncOp = 
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(kernelRefAttr);
      //SmallVector<Value, 8> Loperands(op.getOperands());

      auto Lloc = Lop.getLoc();
      /*
      for(auto arg: operands) {
        //auto kernelArg = getTypeConverter<LLVMTypeConverter>()->promoteOneMemRefDescriptor(
        //  loc, arg, rewriter);

        kernelArgs.push_back(kernelArg);
      }
*/
      auto numKernelOperands = Lop.getNumKernelOperands();

      auto callOperands = op.getOperands();
      for (uint i=0; i<numKernelOperands; i++){
        llArgs.push_back(Lop.getKernelOperand(i));
      }

      SmallVector<Value, 4> kernelArgs;
      kernelArgs.reserve(llArgs.size());

      for (auto it : llvm::zip(callOperands, llArgs)) {
        auto operand = std::get<0>(it);
        auto llvmOperand = std::get<1>(it);

        auto memrefType = operand.getType().dyn_cast<MemRefType>();
        MemRefDescriptor::unpack(rewriter, loc, llvmOperand, memrefType,
                            kernelArgs);
/*
        // For the bare-ptr calling convention, we only have to extract the
        // aligned pointer of a memref.
        if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
          MemRefDescriptor desc(llvmOperand);
          llvmOperand = desc.alignedPtr(rewriter, loc);
        }
        */
        kernelArgs.push_back(llvmOperand);
      }
/*
SmallVector<Value, 4> LLVMTypeConverter::promoteOperands(Location loc,
                                                         ValueRange opOperands,
                                                         ValueRange operands,
                                                         OpBuilder &builder) {
  SmallVector<Value, 4> promotedOperands;
  promotedOperands.reserve(operands.size());
  for (auto it : llvm::zip(opOperands, operands)) {
    auto operand = std::get<0>(it);
    auto llvmOperand = std::get<1>(it);

    if (options.useBarePtrCallConv) {
      // For the bare-ptr calling convention, we only have to extract the
      // aligned pointer of a memref.
      if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
        MemRefDescriptor desc(llvmOperand);
        llvmOperand = desc.alignedPtr(builder, loc);
      } else if (operand.getType().isa<UnrankedMemRefType>()) {
        llvm_unreachable("Unranked memrefs are not supported");
      }
    } else {
      if (operand.getType().isa<UnrankedMemRefType>()) {
        UnrankedMemRefDescriptor::unpack(builder, loc, llvmOperand,
                                         promotedOperands);
        continue;
      }
      if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
        MemRefDescriptor::unpack(builder, loc, llvmOperand, memrefType,
                                 promotedOperands);
        continue;
      }
    }

    promotedOperands.push_back(llvmOperand);
  }
  return promotedOperands;
}
*/
      auto numArguments = kernelArgs.size();
      SmallVector<Type, 4> argumentTypes;
      argumentTypes.reserve(numArguments);
      for (auto argument : kernelArgs) {
        argumentTypes.push_back(argument.getType());
        cobjArgs.push_back(argument);
      }
      /*
      for (uint i = 0; i < numArgs; i++) {
        MemRefDescriptor desc(llvmFuncOp.getOperand(i));
        kernelArgs.push_back(desc);
      }
*/


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
