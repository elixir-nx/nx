cc_binary(
  name = "libexla.so",
  srcs = ["exla.cc"],
  deps = [
    "@org_tensorflow//tensorflow/compiler/xla:literal",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
    "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
    "@org_tensorflow//tensorflow/compiler/xla:shape_util",
    "@org_tensorflow//tensorflow/stream_executor:stream_executor_headers",
    "@org_tensorflow//tensorflow/stream_executor/host:host_platform",
    "@org_tensorflow//tensorflow/core/common_runtime:bfc_allocator",
    # This was linked in after a suggestion from the compiler, but I'm not quite sure right now why it's needed
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
    # TODO: Choose between CPU/GPU depending on what's enabled
    # "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    # TODO: Add condition to check if Exla was built with CUDA
    # "@org_tensorflow//tensorflow/stream_executor/cuda:cuda_platform",
    "@erts//:headers",
  ],
  linkopts = ["-shared"],
  linkshared = True,
)