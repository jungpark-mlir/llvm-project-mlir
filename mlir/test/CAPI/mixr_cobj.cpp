//===- tosa_miir.cpp - Simple test of C and MIIR APIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-full-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/Dialect/Standard.h"
#include "mlir-c/Dialect/Tosa.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/MIGraphX/Pipeline.h"
#include "mlir/Dialect/MIOpen/Pipeline.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitMIOpenDialects.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <string>
#include <vector>

void printToString(MlirStringRef str, void *userData) {
  std::string *strref = static_cast<std::string *>(userData);
  strref->append(str.data, str.length);
}

MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation location) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  // Set func arguments
  int64_t inDims[] = {1, 8, 4, 4};
  int64_t filter0Dims[] = {2, 8, 3, 3};
  int64_t bias0Dims[] = {1};

  MlirType inType = mlirRankedTensorTypeGet(4, inDims, mlirF32TypeGet(ctx),
                                            mlirAttributeGetNull());
  MlirType filter0Type = mlirRankedTensorTypeGet(
      4, filter0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirType bias0Type = mlirRankedTensorTypeGet(
      1, bias0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirType funcBodyArgTypes[] = {inType, filter0Type, bias0Type};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(
      sizeof(funcBodyArgTypes) / sizeof(MlirType), funcBodyArgTypes);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  //-------------- func op

  // Set func attributes
  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString(
               "(tensor<1x8x4x4xf32>, tensor<2x8x3x3xf32>, "
               "tensor<1xf32>) -> (tensor<1x2x2x2xf32>)"));
  MlirAttribute funcNameAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("\"main\""));
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("type")),
          funcTypeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
          funcNameAttr)};

  // Set func op
  MlirOperationState funcState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("builtin.func"), location);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);


/*
//-------------- in0 = migraphx.constant

  // Set constant attributes
  int64_t in0Dims[] = {1, 8, 4, 4};
  float f32in0[128];
  for (int i = 0; i < 128; i++) {
    f32in0[i] = 1.0f;
  }

  MlirAttribute in0ValueAttr = mlirDenseElementsAttrFloatGet(
      mlirRankedTensorTypeGet(4, in0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull()), 128,
      f32in0);
  MlirNamedAttribute in0Attrs[] = {mlirNamedAttributeGet(
      mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")),
      in0ValueAttr)};

  // Set constant op
  MlirType in0Type =
      mlirRankedTensorTypeGet(4, in0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirOperationState in0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("arith.constant"), location);
  mlirOperationStateAddResults(&in0State, 1, &in0Type);
  mlirOperationStateAddAttributes(&in0State, 1, in0Attrs);

  MlirOperation in0Op = mlirOperationCreate(&in0State);
  mlirBlockAppendOwnedOperation(funcBody, in0Op);
  MlirValue in0Value = mlirOperationGetResult(in0Op, 0);


 //-------------- filterc0 = migraphx.constant

  // Set constant attributes
  int64_t filterc0Dims[] = {2, 8, 3, 3};
  float f32filterc0[144];
  for (int i = 0; i < 144; i++) {
    f32filterc0[i] = 1.0f;
  }

  MlirAttribute filterc0ValueAttr = mlirDenseElementsAttrFloatGet(
      mlirRankedTensorTypeGet(4, filterc0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull()), 144,
      f32filterc0);
  MlirNamedAttribute filterc0Attrs[] = {mlirNamedAttributeGet(
      mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")),
      filterc0ValueAttr)};

  // Set constant op
  MlirType filterc0Type =
      mlirRankedTensorTypeGet(4, filterc0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());
  MlirOperationState filterc0State = mlirOperationStateGet(
      mlirStringRefCreateFromCString("arith.constant"), location);
  mlirOperationStateAddResults(&filterc0State, 1, &filterc0Type);
  mlirOperationStateAddAttributes(&filterc0State, 1, filterc0Attrs);

  MlirOperation filterc0Op = mlirOperationCreate(&filterc0State);
  mlirBlockAppendOwnedOperation(funcBody, filterc0Op);
  MlirValue filterc0Value = mlirOperationGetResult(filterc0Op, 0);
*/
  //-------------- conv0 = migraphx.convolution

  // Set conv0 arguments : arg0 from the func and constant filterc0
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue funcArg1 = mlirBlockGetArgument(funcBody, 1);
  MlirValue conv0Operands[] = {funcArg0, funcArg1};

  // Set convolution attributes
  // padding, stride, dilation, group, padding_mode
  MlirAttribute conv0PaddingAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[0:i64, 0:i64, 0:i64, 0:i64]"));
  MlirAttribute conv0StrideAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 1:i64]"));
  MlirAttribute conv0DilationAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString("[1:i64, 1:i64]"));
  MlirAttribute conv0GroupAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("1:i64"));
  MlirAttribute conv0PaddingModeAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("0:i64"));
  MlirNamedAttribute conv0Attrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("padding")),
          conv0PaddingAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("stride")),
          conv0StrideAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dilation")),
          conv0DilationAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("group")),
          conv0GroupAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx,
                            mlirStringRefCreateFromCString("padding_mode")),
          conv0PaddingModeAttr)};

  // Set output shape
  int64_t conv0Dims[] = {1, 2, 2, 2};
  MlirType conv0Type = mlirRankedTensorTypeGet(
      4, conv0Dims, mlirF32TypeGet(ctx), mlirAttributeGetNull());

  // Set convolution op
  MlirOperationState conv0OpState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.convolution"), location);
  mlirOperationStateAddResults(&conv0OpState, 1, &conv0Type);
  mlirOperationStateAddOperands(&conv0OpState, 2, conv0Operands);
  mlirOperationStateAddAttributes(&conv0OpState, 5, conv0Attrs);
  MlirOperation conv0Op = mlirOperationCreate(&conv0OpState);
  mlirBlockAppendOwnedOperation(funcBody, conv0Op);
  MlirValue conv0Value = mlirOperationGetResult(conv0Op, 0);
  /*
    //-------------- migraphx.relu op
    // Set relu0 arguments
    MlirValue relu0Operands[] = {conv0Value};

    // Set relu op
    int64_t relu0Dims[] = {1, 64, 56, 56};
    MlirType relu0Type = mlirRankedTensorTypeGet(4, relu0Dims,
    mlirF32TypeGet(ctx), mlirAttributeGetNull()); MlirOperationState relu0State
    = mlirOperationStateGet( mlirStringRefCreateFromCString("migraphx.relu"),
    location); mlirOperationStateAddResults(&relu0State, 1, &relu0Type);
    mlirOperationStateAddOperands(&relu0State, 1, relu0Operands);

    MlirOperation relu0Op = mlirOperationCreate(&relu0State);
    mlirBlockAppendOwnedOperation(funcBody, relu0Op);
    MlirValue relu0Value = mlirOperationGetResult(relu0Op, 0);
  */
  //-------------- std.return op

  MlirValue retOperands[] = {conv0Value};
  MlirOperationState retState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("std.return"), location);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  MlirOperation module = mlirModuleGetOperation(moduleOp);

  return moduleOp;
}

static bool constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location1 = mlirLocationUnknownGet(ctx);
  MlirModule moduleOp1 = makeAndDumpMIXR(ctx, location1);

  auto module = unwrap(moduleOp1);

  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Initialize LLVM AMDGPU backend.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  mlir::initializeLLVMPasses();

  const char *triple = "amdgcn-amd-amdhsa";
  const char *chip = "gfx908";
  const char *features = "";
  const char *perfConfig = "";

  MlirOperation moduleMO = mlirModuleGetOperation(moduleOp1);

  mlir::PassManager pm(module.getContext(),
                       mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addHighLevelPipeline(pm);
  mlir::miopen::addHighLevelPipeline(pm);

  size_t argIdx = 0;
  module.walk([&](mlir::FuncOp f) {
    auto args = f.getArguments();
    for (auto arg : args) {
      argIdx += 3; // 3 per memref : allocated ptr, aligned ptr, offset
      auto sType = arg.getType().template cast<mlir::ShapedType>();
      auto rank = sType.getRank();
      printf("rank:%d, dim:", rank);
      int i;
      for (i = 0; i < rank; i++)
        printf("<%d>", sType.getDimSize(i));
      printf("\n");
      argIdx += i * 2; // 2 per each dimension : size, stride
    }
    printf("Kernel name : %s\n", f.getName());
  });
  // CHECK: rank:4, dim:<1><64><56><56>
  // CHECK: rank:4, dim:<64><64><1><1>
  // CHECK: rank:1, dim:<64>
  // CHECK: rank:4, dim:<1><64><56><56>
  // CHECK: Kernel name : main

  // 4 memref in this example : input, filter, bias and result
  // example : memref<1x64x56x56xf32>
  // uses 11 params : ptr, ptr, 0 /*offset */, 1, 64, 56, 56, 1, 64, 56, 56
  // printf("Estimated #kernel params : %d\n", argIdx);

  mlir::miopen::addPipeline(pm, perfConfig, false, true);
  mlir::miopen::addBackendPipeline(pm, triple, chip, features);
  auto status = pm.run(module);

  module.walk([&](mlir::LLVM::LLVMFuncOp llvmFunc) {
    size_t block_size =
        llvmFunc->getAttrOfType<mlir::IntegerAttr>("block_size").getInt();
    size_t grid_size =
        llvmFunc->getAttrOfType<mlir::IntegerAttr>("grid_size").getInt();
    auto funcType = llvmFunc.getType().dyn_cast<mlir::LLVM::LLVMFunctionType>();
    int numOperands = funcType.getNumParams();
    printf("kernel params : %d\n", numOperands);
    printf("block_size : %d\n", block_size);
    printf("grid_size : %d\n", grid_size);
  });
  // CHECK: kernel params : 38
  // CHECK: block_size : 64
  // CHECK: grid_size : 56

  size_t size;
  module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      size = hsacoAttr.getValue().size();
      // printf("Binary size : %d\n", size);
    }
  });

  std::vector<char> buffer(size);
  module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    auto hsacoAttr = gpuModule->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (hsacoAttr) {
      std::string hsaco = hsacoAttr.getValue().str();
      std::copy(hsaco.begin(), hsaco.end(), buffer.data());
      /*std::cout << "hsaco = ";
      for(auto o: buffer)
        std::cout << o;
      std::cout << std::endl;*/
    }
  });

  mlirModuleDestroy(moduleOp1);
  if (status.succeeded()) {
    // CHECK: PASSED!
    printf("PASSED!\n");
    return true;
  }
  return false;
}

int main() {
  MlirContext ctx = mlirContextCreate();
  MlirDialectHandle mixrHandle = mlirGetDialectHandle__migraphx__();
  mlirDialectHandleRegisterDialect(mixrHandle, ctx);
  mlirRegisterAllDialects(ctx);

  if (!constructAndTraverseIr(ctx)) {
    printf("FAILED!\n");
    return 1;
  }

  mlirContextDestroy(ctx);

  return 0;
}
