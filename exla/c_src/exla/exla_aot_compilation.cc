#include "tensorflow/compiler/xla/exla/exla_aot_compilation.h"

namespace exla {

  namespace se = tensorflow::se;

  // This is pulled from aot/compile.cc, it uses the
  // Xla aot compiler to compile for CPU, but it looks like
  // it'll support GPU as well
  xla::Status CompileXla(xla::CompileOnlyClient* client,
                         const xla::XlaComputation& computation,
                         const xla::cpu::CpuAotCompilationOptions& aot_opts,
                         tensorflow::tfcompile::CompileResult* compile_result) {

    // Retrieves arg and result layouts from the computation.
    xla::StatusOr<std::unique_ptr<xla::ProgramShape>> pshape_or = client->GetComputationShape(computation);

    if (!pshape_or.ok()) {
      return tensorflow::errors::Unknown("Couldn't get XLA program shape: ", pshape_or.status().error_message());
    }

    compile_result->program_shape = pshape_or.ValueOrDie()->ToProto();
    xla::ProgramShapeProto* pshape = &compile_result->program_shape;

    // AotXlaComputationInstance::argument_layouts is a vector of Shape
    // pointers. Accumulate the Shape objects themselves in a separate vector
    // while building the vector of pointers.
    std::vector<const xla::Shape*> arg_layout_ptrs(pshape->parameters_size());
    std::vector<xla::Shape> arg_layouts(pshape->parameters_size());

    for (int i = 0; i < pshape->parameters_size(); ++i) {
      arg_layouts[i] = xla::Shape(*pshape->mutable_parameters(i));
      arg_layout_ptrs[i] = &arg_layouts[i];
    }

    xla::CompileOnlyClient::AotXlaComputationInstance instance;
    instance.computation = &computation;
    instance.argument_layouts = std::move(arg_layout_ptrs);
    xla::Shape result_shape(pshape->result());
    instance.result_layout = &result_shape;
    xla::StatusOr<std::vector<std::unique_ptr<xla::AotCompilationResult>>> aot_or = client->CompileAheadOfTime({instance}, aot_opts);

    if (!aot_or.ok()) {
      return tensorflow::errors::Unknown("XLA compilation failed: ", aot_or.status().error_message());
    }

    compile_result->aot = xla::unique_ptr_static_cast<xla::cpu::CpuAotCompilationResult>(std::move(aot_or.ValueOrDie().back()));
    compile_result->entry_point = aot_opts.entry_point_name();
    compile_result->pointer_size = xla::CompileOnlyClient::PointerSizeForTriple(aot_opts.triple());

    return tensorflow::Status::OK();
  }

  xla::Status CompileComputation(const xla::XlaComputation& computation,
                                 std::string pbtext_path,
                                 std::string header_path,
                                 std::string object_path,
                                 std::string function_name,
                                 std::string class_name,
                                 std::string target_triple,
                                 std::string target_features) {

    xla::Status compilation_status;

    se::Platform* cpu_platform = xla::PlatformUtil::GetPlatform("Host").ConsumeValueOrDie();
    xla::CompileOnlyClient* client = xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform).ValueOrDie();

    // Read the generated protobuf input, we can do it as a file, or pass it as a string/binary
    tensorflow::tf2xla::Config config;
    compilation_status.Update(tensorflow::ReadTextProto(tensorflow::Env::Default(), pbtext_path, &config));

    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    std::string entry_point = absl::StrCat("exla_", function_name, "_entry_point");

    // These options are flags we can give to the user
    xla::cpu::CpuAotCompilationOptions aot_opts(
      target_triple,
      /*target_cpu=*/"",
      target_features,
      entry_point,
      xla::cpu::CpuAotCompilationOptions::RelocationModel::BigPic
    );

    // From aot/compile, a struct which holds the aot compilation
    tensorflow::tfcompile::CompileResult compile_result;

    // Compile the Xla computation and populate compile_result
    compilation_status.Update(CompileXla(client, computation, aot_opts, &compile_result));

    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    // This is an object file
    const std::vector<char>& obj = compile_result.aot->object_file_data();

    // Write it to a file
    compilation_status.Update(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                                            object_path,
                                                            absl::string_view(obj.data(), obj.size())));
    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    tensorflow::tfcompile::CodegenOpts codegen_opts;
    codegen_opts.class_name = class_name;
    codegen_opts.gen_name_to_index = false;
    codegen_opts.gen_program_shape = false;
    codegen_opts.gen_hlo_profile_printer_data = false;
    codegen_opts.target_triple = target_triple;

    compilation_status.Update(tensorflow::tfcompile::ParseCppClass(class_name,
                                                                   &codegen_opts.class_name,
                                                                   &codegen_opts.namespaces));
    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    tensorflow::tfcompile::MetadataResult metadata_result;
    compilation_status.Update(tensorflow::tfcompile::GenerateMetadata(codegen_opts,
                                                                      compile_result,
                                                                      &metadata_result));

    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    // The header file
    std::string header;
    compilation_status.Update(tensorflow::tfcompile::GenerateHeader(codegen_opts,
                                                                    config,
                                                                    compile_result,
                                                                    metadata_result,
                                                                    &header));
    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    // Write Header to file
    compilation_status.Update(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                                            header_path,
                                                            header));
    if (!compilation_status.ok()) {
      LOG(ERROR) << compilation_status;
      return compilation_status;
    }

    return compilation_status;
  }
}