#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/aot/codegen.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"


namespace exla{

  xla::Status CompileXla(xla::CompileOnlyClient* client,
                         const xla::XlaComputation& computation,
                         const xla::cpu::CpuAotCompilationOptions& aot_opts,
                         tensorflow::tfcompile::CompileResult* compile_result);

  xla::Status CompileComputation(const xla::XlaComputation& computation,
                                 std::string pbtext_path,
                                 std::string header_path,
                                 std::string object_path,
                                 std::string function_name,
                                 std::string class_name,
                                 std::string target_triple,
                                 std::string target_features);

}