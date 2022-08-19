//===- MIGraphX.cpp - C Interface for MIGraphX dialect
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/Pipeline.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include <mutex>
#include <vector>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx,
                                      mlir::migraphx::MIGraphXDialect)

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED
void mlirGetKernelInfo(MlirModule module, int *size, void *data) {
  auto mod = unwrap(module);
  int argNum = 0;
  int argIdx = 0;
  llvm::StringRef kernelName;

  // Either of pointers should be provided.
  assert((size != nullptr || data != nullptr) &&
         "Either size or data pointer should be provided");
  std::vector<int> info;
  mod.walk([&](mlir::func::FuncOp f) {
    auto args = f.getArguments();
    for (auto arg : args) {
      argNum++;
      auto sType = arg.getType().template cast<mlir::ShapedType>();
      auto rank = sType.getRank();
      info.push_back(rank);
      for (int i = 0; i < rank; i++)
        info.push_back(sType.getDimSize(i));
      argIdx += rank;
    }
    kernelName = f.getName();
  });
  if (data == nullptr && size != nullptr) {
    *size = (1 + argNum + argIdx) * sizeof(int) + kernelName.size();
  } else if (data != nullptr) {
    int argSize = argNum + argIdx;
    int *argData = (int *)data;
    argData[0] = argNum;
    for (int i = 0; i < argSize; i++)
      argData[i + 1] = info[i];
    char *nameData = (char *)(argData + argSize + 1);
    for (size_t i = 0, e = kernelName.size(); i < e; ++i) {
      nameData[i] = kernelName[i];
    }
  }
}

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs) {
  auto mod = unwrap(module);
  mod.walk([&](mlir::LLVM::LLVMFuncOp llvmFunc) {
    attrs[0] =
        llvmFunc->getAttrOfType<mlir::IntegerAttr>("block_size").getInt();
    attrs[1] = llvmFunc->getAttrOfType<mlir::IntegerAttr>("grid_size").getInt();
  });
}

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, int *size, char *bin) {
  bool success = false;
  auto mod = unwrap(module);
  if (bin == nullptr && size == nullptr)
    return success;
  mod.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      if (bin != nullptr) { // return binary regardless the presence of *size
        std::string hsaco = hsacoAttr.getValue().str();
        std::copy(hsaco.begin(), hsaco.end(), bin);
        success = true;
      } else {
        *size = hsacoAttr.getValue().size();
      }
    }
  });
  return success;
}

//
std::string stringfyConfig(int32_t numParams, uint32_t *perfConfig) {
  std::string result;
  int i = 0;
  for (; i < numParams - 1; i++){
    result.append(perfConfig[i]);
    result.append(',');
  }
  result.append(perfConfig[i]);
  return result;
}

void getRealConv2dPerfConfig(MlirMIOpenOpDataType dType, int32_t nParam, uint32_t *pConfig) {
    // bool GemmBThreadCopyMoreGemmKPack
    // bool GemmAThreadCopyMoreGemmK
    // <4, 256> GemmMPerBlock
    constexpr std::vector<uint32_t> p2 = {4, 8, 16, 32, 64, 128, 256};
    // <16, 256> GemmNPerBlock
    constexpr std::vector<uint32_t> p3 = {16, 32, 64, 128, 256};
    // <4, 128> GemmMPerWave
    constexpr std::vector<uint32_t> p4 = {4, 8, 16, 32, 64, 128};
    // <16, 128> GemmNPerWave
    constexpr std::vector<uint32_t> p5 = {16, 32, 64, 128};
    // <4, 8> GemmKPACKSize
    constexpr std::vector<uint32_t> p6 = {4, 8};
    // INT8<8, 32> or <1, 8> Gem1mKPerBlock
    constexpr std::vector<uint32_t> p7a = {8, 16, 32};
    constexpr std::vector<uint32_t> p7b = {1, 2, 4, 8};

    // Replace the index with the real value.
    pConfig[2] = p2[pConfig[2]];
    pConfig[3] = p3[pConfig[3]];
    pConfig[4] = p4[pConfig[4]];
    pConfig[5] = p5[pConfig[5]];
    pConfig[6] = p6[pConfig[6]];
    if (dType == INT32)
      pConfig[7] = p7a[pConfig[7]];
    else
      pConfig[7] = p7b[pConfig[7]];
}

// Traverse tunable ops and update perf_config attribute
MLIR_CAPI_EXPORTED bool mlirSetTuningParams(MlirModule module,
                                            MlirMIOpenTunableOpConfig problem,
                                            void *perfConfig, bool bPerfStr) {
  if (!perfConfig)
    return false; // fail as nullptr passed.
  int32_t numParams;
  bool bFound = false;
  auto mod = unwrap(module);
  mod.walk([&](mlir::miopen::Conv2DOp convOp) {
    if (bFound)
      return false; // fail because the module contains multiple tunable ops.
    bFound = true;
    if (problem.opType != conv2d)
      return false; // invalid op requested.
    ImplicitLocOpBuilder b(convOp->getLoc(), convOp->getContext());
    numParams = getNumTuningParams(problem);
    if (bPerfStr) { // set string attr
      StringAttr attr = b.getStringAttr(static_cast<std::string*>(perfConfig));
      convOp->setAttr("perf_config", attr);
    }
    else {
      // get real values first.
      getRealConv2dPerfConfig(numParams, perfConfig, problem.convCfg.data_type);
      // stringfy and set attribute
      std::string perfString = stringfyConfig(numParams, perfConfig);
      StringAttr attr = b.getStringAttr(perfString);
      convOp->setAttr("perf_config", attr);
    }
  });

  mod.walk([&](mlir::miopen::GemmOp gemmOp) {
    return false; // fail, unsupported yet
  });

  return true; // succeed.
}

// returns information so user can create a tuning space, OwnedRanges
MLIR_CAPI_EXPORTED int32_t mlirGetTuningInfo(MlirModule module,
                                             MlirMIOpenTunableOpConfig *problem,
                                             uint32_t *tuningRanges) {
  if (!problem || !tuningRanges)
    return 0; // fail as nullptr passes for the tuning.

  int32_t numParams;
  bool bFound = false;
  auto mod = unwrap(module);

  // miopen.conv2d
  mod.walk([&](mlir::miopen::Conv2DOp convOp) {
    if (bFound)
      return 0; // fail because the module contains multiple tunable ops.
    bFound = true;
    problem.opType = conv2d;
    problem.convCfg.direction = FWD;

    // Extract convolution config.
    StringRef inLayout =
        convOp->getAttrOfType<StringAttr>("input_layout").getValue();
    StringRef filterLayout =
        convOp->getAttrOfType<StringAttr>("filter_layout").getValue();
    StringRef outputLayout =
        convOp->getAttrOfType<StringAttr>("output_layout").getValue();

    // Only expect these two combinations for now.
    StringRef iNCHW = "nchwg";
    StringRef fNCHW = "kcyxg";
    StringRef oNCHW = "nkhwg";
    StringRef iNHWC = "nhwcg";
    StringRef fNHWC = "kyxcg";
    StringRef oNHWC = "nhwkg";

    if (inLayout.compare(iNCHW) == 0) {
      if (filterLayout.compare(fNCHW) != 0 || outputLayout.compare(oNCHW) != 0)
        return 0; // unexpected layout, failing.
      problem.convCfg.layout = NCHW;
    } else if (inLayout.compare(iNHWC) == 0) {
      if (filterLayout.compare(fNHWC) != 0 || outputLayout.compare(oNHWC) != 0)
        return 0; // unexpected layout, failing.
      problem.convCfg.layout = NHWC;
    } else {
      return 0; // unexpected layout, failing.
    }

    auto input = convOp->getOperand(0);
    auto weight = convOp->getOperand(1);
    ShapedType inTy = input.getType().cast<ShapedType>();
    auto inShape = inTy.getShape();
    problem.convCfg.batchsize = inShape[0];

    auto elemTy = inTy.getElementType();
    if (elemTy.isF32())
      problem.convCfg.data_type = FP32;
    else if (elemTy.isF16())
      problem.convCfg.data_type = FP16;
    else if (elemTy.isBF16())
      problem.convCfg.data_type = BF16;
    else if (elemTy.isInteger(32))
      problem.convCfg.data_type = INT32;
    else if (elemTy.isInteger(8))
      problem.convCfg.data_type = INT8;
    else
      return 0; // unsupported type, failing.

    ShapedType wtTy = weight.getType().cast<ShapedType>();
    auto wtShape = wtTy.getShape();
    problem.convCfg.out_channels = wtShape[0];

    if (problem.convCfg.layout == NCHW) {
      problem.convCfg.in_channels = inShape[1];
      problem.convCfg.in_h = inShape[2];
      problem.convCfg.in_w = inShape[3];
      problem.convCfg.f_h = wtShape[2];
      problem.convCfg.f_w = wtShape[3];
    } else { // NHWC
      problem.convCfg.in_channels = inShape[3];
      problem.convCfg.in_h = inShape[2];
      problem.convCfg.in_w = inShape[3];
      problem.convCfg.f_h = wtShape[2];
      problem.convCfg.f_w = wtShape[3];
    }

    auto pad = convOp.pad();
    auto stride = convOp.stride();
    auto dilation = convOp.dilation();

    // pad top and left instead.
    problem.convCfg.pad_h = pad[0].dyn_cast<IntegerAttr>().getInt();
    problem.convCfg.pad_w = pad[2].dyn_cast<IntegerAttr>().getInt();
    problem.convCfg.conv_stride_h = stride[0].dyn_cast<IntegerAttr>().getInt();
    problem.convCfg.conv_stride_w = stride[1].dyn_cast<IntegerAttr>().getInt();
    problem.convCfg.dilation_h = dilation[0].dyn_cast<IntegerAttr>().getInt();
    problem.convCfg.dilation_w = dilation[1].dyn_cast<IntegerAttr>().getInt();

    problem.convCfg.group_count = inShape[4];

    numParams = getNumTuningParams(problem);
    // bool GemmBThreadCopyMoreGemmKPack
    // bool GemmAThreadCopyMoreGemmK
    // <4, 256> GemmMPerBlock
    // <16, 256> GemmNPerBlock
    // <4, 128> GemmMPerWave
    // <16, 128> GemmNPerWave
    // <4, 8> GemmKPACKSize
    // INT8<8, 32> or <1, 8> Gem1mKPerBlock
    std::vector<uint32_t> ranges = {2, 2, 7, 5, 6, 4, 2, 4};
    if (problem.convCfg.data_type == INT8)
      ranges[7] = 3;
    tuningRanges = ranges.data();
  });

  // miopen.gemm
  mod.walk([&](mlir::miopen::GemmOp gemmOp) {
    if (bFound)
      return 0; // fail because the module contains multiple tunable ops.
    bFound = true;
    numParams = 0; // Not supported yet
  });

  return numParams;
}

// pipelines

MLIR_CAPI_EXPORTED
void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm) {
  auto passMan = unwrap(pm);
  // FIXME : WA for the multithreading issue, potentially fixed in upstream.
  passMan->getContext()->disableMultithreading();
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addHighLevelPipeline(*passMan);
  mlir::miopen::buildBufferizePipeline(*passMan);
}

MLIR_CAPI_EXPORTED void mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *chip,
                                                       const char *triple,
                                                       const char *features) {
  static std::mutex target_mutex;
  target_mutex.lock();
  // Some calls included in regiserGpuSerializeToHsacoPass() are not thread safe
  // and user may call this pipeline from different threads.
  mlir::registerGpuSerializeToHsacoPass();
  target_mutex.unlock();
  auto passMan = unwrap(pm);
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::miopen::KernelOptions kOpts;
  kOpts.tuningFallback = true;
  mlir::miopen::buildKernelPipeline(*passMan, kOpts);
  mlir::miopen::BackendOptions opts;
  opts.triple = triple;
  opts.chip = chip;
  opts.features = features;
  opts.optLevel = 3;
  opts.indexBitwidth = 64;
  mlir::miopen::buildBackendPipeline(*passMan, opts);
}
