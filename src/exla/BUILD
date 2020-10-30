cc_binary(
  name = "libbuilder.so",
  srcs = ["builder.cc"],
  deps = [
    "@com_google_absl//absl/memory",
    "@org_tensorflow//tensorflow/compiler/xla:literal",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
    "@erts//:headers",
  ],
  linkopts = ["-shared"],
  linkshared = True,
)