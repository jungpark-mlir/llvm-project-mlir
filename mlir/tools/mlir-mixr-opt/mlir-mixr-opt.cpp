//===- mlir-mixr-opt.cpp - The MLIR MIGraphX opt ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MIXR codegen tool
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Reducer/OptReductionPass.h"
#include "mlir/Reducer/Passes/OpReducer.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionTreePass.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace mlir {
namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test
} // namespace mlir

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::Required,
                                                llvm::cl::desc("<input file>"));

static llvm::cl::opt<std::string>
    testFilename("test", llvm::cl::Required, llvm::cl::desc("Testing script"));

static llvm::cl::list<std::string>
    testArguments("test-args", llvm::cl::ZeroOrMore,
                  llvm::cl::desc("Testing script arguments"));

static llvm::cl::opt<std::string>
    outputFilename("o",
                   llvm::cl::desc("Output filename for the reduced test case"),
                   llvm::cl::init("-"));

// TODO: Use PassPipelineCLParser to define pass pieplines in the command line.
static llvm::cl::opt<std::string>
    passTestSpecifier("pass-test",
                      llvm::cl::desc("Indicate a specific pass to be tested"));

// Parse and verify the input MLIR file.
static LogicalResult loadModule(MLIRContext &context, OwningModuleRef &module,
                                StringRef inputFilename) {
  module = parseSourceFile(inputFilename, &context);
  if (!module)
    return failure();

  return success();
}

int main(int argc, char **argv) {

  llvm::InitLLVM y(argc, argv);

  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR mixr opt.\n");

  std::string errorMessage;

  auto testscript = openInputFile(testFilename, &errorMessage);
  if (!testscript)
    llvm::report_fatal_error(errorMessage);

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output)
    llvm::report_fatal_error(errorMessage);

  mlir::MLIRContext context;
  registerAllDialects(context.getDialectRegistry());

  mlir::OwningModuleRef moduleRef;
  if (failed(loadModule(context, moduleRef, inputFilename)))
    llvm::report_fatal_error("Input test case can't be parsed");

  // Initialize test environment.
  const Tester test(testFilename, testArguments);

  PassManager pm(&context);

  pm.addPass(std::make_unique<OptReductionPass>(test, &context,
                                                createGPUToMIGraphX()));

  ModuleOp m = moduleRef.get().clone();

  if (failed(pm.run(m)))
    llvm::report_fatal_error("Error running the reduction pass pipeline");

  m.print(output->os());
  output->keep();

  return 0;
}
