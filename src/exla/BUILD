load("@com_grail_bazel_compdb//:aspects.bzl", "compilation_database")
load("@org_tensorflow//tensorflow/core/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured",)

package(default_visibility=["//visibility:public"])

compilation_database(
  name = "exla_compdb",
  targets = [
      ":libexla_cpu.so",
      ":libexla_gpu.so",
  ],
)

cc_library(
  name = "exla_allocator",
  srcs = ["exla_allocator.cc"],
  hdrs = ["exla_allocator.h"],
  deps = [
    "@org_tensorflow//tensorflow/stream_executor:stream_executor_headers",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
  ],
)

# I don't like the NIF dependency
cc_library(
  name = "exla_macros",
  hdrs = ["exla_macros.h"],
  deps = [
    ":exla_nif_util"
  ]
)

cc_library(
  name = "exla_nif_util",
  srcs = ["exla_nif_util.cc"],
  hdrs = ["exla_nif_util.h"],
  deps = [
    "@com_google_absl//absl/types:span",
    "@org_tensorflow//tensorflow/stream_executor:stream_executor_headers",
    "@org_tensorflow//tensorflow/compiler/xla:executable_run_options",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:executable_build_options",
    "@erts//:headers"
  ],
)

# This target links in both GPU/CPU Device
cc_binary(
  name = "libexla_gpu.so",
  srcs = ["exla.cc"],
  deps = if_cuda_is_configured([
    "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    ]) + [
    ":exla_allocator",
    ":exla_nif_util",
    ":exla_macros",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)

# This target links in just CPU device
cc_binary(
  name = "libexla_cpu.so",
  srcs = ["exla.cc"],
  deps = [
    ":exla_allocator",
    ":exla_nif_util",
    ":exla_macros",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)