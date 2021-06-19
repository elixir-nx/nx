load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_cuda_or_rocm",)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_with_tpu_support",)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency",)

package(default_visibility=["//visibility:private"])

cc_library(
  name = "exla_log_sink",
  hdrs = ["exla_log_sink.h"],
  deps = [
    ":exla_nif_util",
    "@org_tensorflow//tensorflow/core:lib",
  ],
)

cc_library(
  name = "exla_client",
  srcs = ["exla_client.cc"],
  hdrs = ["exla_client.h"],
  deps = [
    ":exla_nif_util",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed:protocol",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed:protocol_proto_cc",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed:client",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_stream_executor_client",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:local_device_state",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:event_pool",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:semaphore",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:worker_thread",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:tracked_device_buffer",
    "@org_tensorflow//tensorflow/core/common_runtime:bfc_allocator",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:tpu_client",
     tf_grpc_cc_dependency(),
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
  ] + if_with_tpu_support([
    "@org_tensorflow//tensorflow/core/tpu:tpu_executor_dlsym_initializer",
  ]),
  linkopts = ["-shared"],
  linkshared = 1,
)
