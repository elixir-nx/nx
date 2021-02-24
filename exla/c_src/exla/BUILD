load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_cuda_or_rocm",)

package(default_visibility=["//visibility:private"])

cc_library(
  name = "exla_allocator",
  srcs = ["exla_allocator.cc"],
  hdrs = ["exla_allocator.h"],
  deps = [
    ":exla_nif_util",
    ":exla_device",
    "@org_tensorflow//tensorflow/core:lib",
    "@org_tensorflow//tensorflow/core/framework:allocator",
    "@org_tensorflow//tensorflow/core/common_runtime:bfc_allocator",
    "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_bfc_allocator",
    "@org_tensorflow//tensorflow/core/common_runtime/device:device_mem_allocator",
    "@org_tensorflow//tensorflow/stream_executor:tf_allocator_adapter",
  ],
)

cc_library(
  name = "exla_log_sink",
  hdrs = ["exla_log_sink.h"],
  deps = [
    ":exla_nif_util",
    "@org_tensorflow//tensorflow/core:lib",
  ],
)

cc_library (
  name = "exla_device",
  srcs = ["exla_device.cc"],
  hdrs = ["exla_device.h"],
  deps = [
    "@org_tensorflow//tensorflow/compiler/xla/client:local_client",
    "@org_tensorflow//tensorflow/stream_executor:stream_executor",
  ],
)

cc_library(
  name = "exla_client",
  srcs = ["exla_client.cc"],
  hdrs = ["exla_client.h"],
  deps = [
    ":exla_device",
    ":exla_allocator",
    ":exla_nif_util",
    "@org_tensorflow//tensorflow/core/framework:allocator",
    "@org_tensorflow//tensorflow/compiler/xla:cpu_function_runtime",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
    "@org_tensorflow//tensorflow/core:lib",
  ],
)

cc_library(
  name = "exla_nif_util",
  srcs = ["exla_nif_util.cc"],
  hdrs = ["exla_nif_util.h"] + glob(["erts/**/*.h"]),
  deps = [
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/core:lib",
  ],
)

cc_library(
  name = "exla_aot_compilation",
  srcs = ["exla_aot_compilation.cc"],
  hdrs = ["exla_aot_compilation.h"],
  deps = [
    "@org_tensorflow//tensorflow/compiler/xla/client:compile_only_client",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla/service/cpu:cpu_compiler",
    "@org_tensorflow//tensorflow/compiler/aot:tfcompile_main",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
    "@org_tensorflow//tensorflow/compiler/xla:util",
    "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
    "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla",
  ],
)

cc_binary(
  name = "libexla.so",
  srcs = ["exla.cc"],
  deps = if_cuda_or_rocm([
    "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    ]) + [
    ":exla_nif_util",
    ":exla_client",
    ":exla_aot_compilation",
    ":exla_log_sink",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client/lib:lu_decomposition",
    "@org_tensorflow//tensorflow/compiler/xla/client/lib:math",
    "@org_tensorflow//tensorflow/compiler/xla/client/lib:qr",
    "@org_tensorflow//tensorflow/compiler/xla/client/lib:svd",
    "@org_tensorflow//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
    "@org_tensorflow//tensorflow/compiler/xla:comparison_util",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)