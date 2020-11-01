cc_binary(
  name = "libexla.so",
  srcs = ["exla.cc"],
  deps = [
    "@com_google_absl//absl/memory",
    "@org_tensorflow//tensorflow/compiler/xla:literal",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
    "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
    # This was linked in after a suggestion from the compiler, but I'm not quite sure right now why it's needed
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
    # TODO: Add condition to check if Exla was built with CUDA
    "@org_tensorflow//tensorflow/stream_executor/cuda:cuda_platform",
    "@erts//:headers",
  ],
  linkopts = ["-shared"],
  linkshared = True,
)