//===-- mlir-c/Dialect/MIGraphX.h - C API for MIGraphX dialect --------*- C
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_MIGRAPHX_H
#define MLIR_C_DIALECT_MIGRAPHX_H

#include "mlir-c/Pass.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirMIOpenConvLayout { NCHW, NHWC };
enum MlirMIOpenOpDataType { FP32, FP16, BF16, INT8, INT32 };
enum MlirMIOpenConvDirection { FWD, BWD, WT };
enum MlirMIOpenTunableOpType { conv2d, gemm };

// Problem config struct, using existing struct from the MIOpen tuning DB
struct conv2dConfig {
  MlirMIOpenConvLayout layout;
  MlirMIOpenOpDataType data_type;
  MlirMIOpenConvDirection direction;
  // spatial_dim
  uint64_t in_channels;
  uint64_t in_h;
  uint64_t in_w;
  // in_d
  uint64_t fil_h;
  uint64_t fil_w;
  // fil_d
  uint64_t out_channels;
  uint64_t batchsize;
  uint64_t pad_h;
  uint64_t pad_w;
  // pad_d
  uint64_t conv_stride_h;
  uint64_t conv_stride_w;
  // conv_stride_d
  uint64_t dilation_h;
  uint64_t dilation_w;
  // dilation_d
  // bias
  uint64_t group_count;
};

struct gemmConfig {
  uint64_t M;
  uint64_t K;
  uint64_t N;
};

union MlirMIOpenTunableOpConfig {
  MlirMIOpenTunableOpType opType;
  struct conv2dConfig convCfg;
  struct gemmConfig gemmCfg;
};

static int32_t getNumTuningParams(union MlirMIOpenTunableOpConfig op){
  if (op.opType == conv2d)
    return 8;
  else
    return 0; // unsupported.
}

    MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx);

// Phase 0 functions : Assuming the given module contains only one function

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED void mlirGetKernelInfo(MlirModule module, int *size,
                                          void *data);

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs);

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, int *size, char *bin);

// Find tunable op and update perf_config attribute
MLIR_CAPI_EXPORTED bool mlirSetTuningParams(MlirModule module,
                                            MlirMIOpenTunableOpConfig problem,
                                            void *perfConfig, bool bPerfStr);

// Find tunable op and get tuning information
// returns the number of tuning parameters
// returns the range of each parameters in tuningRanges
MLIR_CAPI_EXPORTED unsigned
mlirGetTuningInfo(MlirModule module, MlirMIOpenTunableOpConfig *problem,
                  unsigned *tuningRanges);

// pipelines

MLIR_CAPI_EXPORTED void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm);

MLIR_CAPI_EXPORTED void mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *chip,
                                                       const char *triple,
                                                       const char *features);
#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_MIGRAPHX_H
