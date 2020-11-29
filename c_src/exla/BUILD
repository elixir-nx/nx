load("@org_tensorflow//tensorflow/core/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured",)

package(default_visibility=["//visibility:private"])

cc_library(
  name = "exla_allocator",
  srcs = ["exla_allocator.cc"],
  hdrs = ["exla_allocator.h"] + glob(["erts/**/*.h"]),
  deps = [
    "@org_tensorflow//tensorflow/core/framework:allocator",
    "@org_tensorflow//tensorflow/core:lib",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
  ],
)

cc_library (
  name = "exla_device",
  srcs = ["exla_device.cc"],
  hdrs = ["exla_device.h"],
  deps = [
    "@org_tensorflow//tensorflow/compiler/xla/client:local_client",
    "@org_tensorflow//tensorflow/stream_executor:stream_executor",
    "@com_google_absl//absl/memory",
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
    "@org_tensorflow//tensorflow/compiler/xla:cpu_function_runtime",
    "@org_tensorflow//tensorflow/compiler/xla:literal_util",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
    "@org_tensorflow//tensorflow/core:lib",
    "@org_tensorflow//tensorflow/core/framework:allocator",
    "@org_tensorflow//tensorflow/core/common_runtime:bfc_allocator",
    "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_bfc_allocator",
    "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_mem_allocator",
    "@org_tensorflow//tensorflow/stream_executor:tf_allocator_adapter",
    "@com_google_absl//absl/memory",
  ],
)

cc_library(
  name = "exla_nif_util",
  srcs = ["exla_nif_util.cc"],
  hdrs = ["exla_nif_util.h"] + glob(["erts/**/*.h"]),
  deps = [
    ":exla_allocator",
    "@com_google_absl//absl/types:span",
    "@org_tensorflow//tensorflow/stream_executor:stream_executor_headers",
    "@org_tensorflow//tensorflow/compiler/xla:executable_run_options",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:executable_build_options",
    "@org_tensorflow//tensorflow/core:lib",
  ],
)

cc_binary(
  name = "libexla_cuda.so",
  srcs = ["exla.cc"],
  deps = if_cuda_is_configured([
    "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    ]) + [
    ":exla_allocator",
    ":exla_nif_util",
    ":exla_client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla:comparison_util",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)

# This target links in just CPU device
cc_binary(
  name = "libexla_host.so",
  srcs = ["exla.cc"],
  deps = [
    ":exla_allocator",
    ":exla_nif_util",
    ":exla_client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla:comparison_util",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)
