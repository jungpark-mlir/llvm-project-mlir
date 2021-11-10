add_mlir_dialect_library(MLIRMIGraphXPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  LLVMROCmBackendUtils
  MLIRGPUToROCDLTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRStandard
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRSCFToStandard
  MLIRSupport
  MLIRTransformUtils
)
