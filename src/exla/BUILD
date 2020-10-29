# Bazel has a py_binary rule that builds python modules automatically with deps...
# Might be ideal to create something similar for Elixir, but this workaround is suitable for now.
cc_binary(
  name = "libexla.so",
  srcs = ["exla.cc"],
  copts = ["-Iexternal/erts"],
  deps = [
    "@com_google_absl//absl/algorithm:container",
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/container:flat_hash_set",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/types:span",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@erts//:headers",
  ],
  linkopts = ["-shared"],
  linkshared = True,
)