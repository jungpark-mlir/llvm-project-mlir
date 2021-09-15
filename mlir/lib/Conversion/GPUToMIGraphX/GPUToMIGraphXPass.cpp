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
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"
#include "mlir/ExecutionEngine/OptUtils.h"


using namespace mlir;

struct GPUToMIGraphX
    : public GPUToMIGraphXBase<GPUToMIGraphX> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<migraphx::MIGraphXDialect, StandardOpsDialect, gpu::GPUDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<migraphx::MIGraphXDialect, StandardOpsDialect, gpu::GPUDialect>();
    target.addIllegalOp<CallOp>();
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

std::unique_ptr<Pass> migraphx::createGPUToMIGraphX() {
  return std::make_unique<GPUToMIGraphX>();
}
void migraphx::addGPUToMIGraphXPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createGPUToMIGraphX());

  //applyPassManagerCLOptions(pm);
  static llvm::cl::opt<std::string> tripleName("triple", llvm::cl::desc("target triple"),
                                       llvm::cl::value_desc("triple string"),
                                       llvm::cl::init(""));

  static llvm::cl::opt<std::string> targetChip("target", llvm::cl::desc("target chip"),
                                       llvm::cl::value_desc("AMDGPU ISA version"),
                                       llvm::cl::init(""));

  static llvm::cl::opt<std::string> features("feature", llvm::cl::desc("target features"),
                                     llvm::cl::value_desc("AMDGPU target features"),
                                     llvm::cl::init(""));

  bool systemOverride = false;
  if (tripleName.empty() && targetChip.empty() && features.empty()) {
    systemOverride = true;
  }
  BackendUtils utils(tripleName, targetChip, features, systemOverride);

  const char gpuBinaryAnnotation[] = "rocdl.hsaco";


  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
  kernelPm.addPass(createConvertGPUKernelToBlobPass(
      [&utils](Operation *m, llvm::LLVMContext &llvmContext,
               llvm::StringRef name) {
        return utils.compileModuleToROCDLIR(m, llvmContext, name);
      },
      [&utils](const std::string isa, Location loc, StringRef name) {
        return utils.compileISAToHsaco(isa, loc, name);
      },
      utils.getTriple(), utils.getChip(), utils.getFeatures(),
      gpuBinaryAnnotation));
}
