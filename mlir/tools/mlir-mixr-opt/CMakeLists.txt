get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AsmParser
  Core
  IRReader
  Support
  Target
  TransformUtils
  )

set(LIBS
  ${dialect_libs}
  ${conversion_libs}

  LLVMX86AsmParser
  LLVMAMDGPUAsmParser
  LLVMROCmBackendUtils
  MLIRExecutionEngine
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIRIR
  MLIRLoopAnalysis
  MLIROptLib

  MLIRPass

  MLIRTransforms
  MLIRTransformUtils

  MLIRParser
  MLIRROCDLIR
  MLIRSupport
  MLIRTargetLLVMIR
  MLIRTargetROCDLIR
  )

add_llvm_tool(mlir-mixr-opt
  mlir-mixr-opt.cpp
  )

target_link_libraries(mlir-mixr-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-mixr-opt)

mlir_check_all_link_libraries(mlir-mixr-opt)
