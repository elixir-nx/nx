load("@com_grail_bazel_compdb//:aspects.bzl", "compilation_database")

compilation_database(
    name = "exla_compdb",
    targets = [
        ":libexla.so",
    ],
)

cc_library(
  name = "exla_allocator",
  srcs = ["exla_allocator.cc"],
  hdrs = ["exla_allocator.h"],
  deps = [
    "@org_tensorflow//tensorflow/stream_executor:stream_executor_headers",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
  ],)

cc_binary(
  name = "libexla.so",
  srcs = ["exla.cc"],
  deps = [
    ":exla_allocator",
    "@org_tensorflow//tensorflow/compiler/xla:literal",
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
    "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
    "@org_tensorflow//tensorflow/compiler/xla:shape_util",
    "@org_tensorflow//tensorflow/stream_executor:stream_executor_headers",
    # This was linked in after a suggestion from the compiler, but I'm not quite sure right now why it's needed
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
    # TODO: Choose between CPU/GPU depending on what's enabled
    "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    "@erts//:headers",
  ],
  linkopts = ["-shared"],
  linkshared = True,
)