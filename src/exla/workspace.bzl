load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def exla_workspace():

  tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
  )

  # gRPC wants the existence of a cares dependence but its contents are not
  # actually important since we have set GRPC_ARES=0 in tools/bazel.rc
  native.bind(
    name = "cares",
    actual = "@grpc//third_party/nanopb:nanopb",
  )

  # ===== Pin `com_google_absl` with the same version(and patch) with Tensorflow.
  tf_http_archive(
    name = "com_google_absl",
    build_file = str(Label("@org_tensorflow//third_party:com_google_absl.BUILD")),
    # TODO: Remove the patch when https://github.com/abseil/abseil-cpp/issues/326 is resolved
    # and when TensorFlow is build against CUDA 10.2
    patch_file = str(Label("@org_tensorflow//third_party:com_google_absl_fix_mac_and_nvcc_build.patch")),
    sha256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",  # SHARED_ABSL_SHA
    strip_prefix = "abseil-cpp-df3ea785d8c30a9503321a3d35ee7d35808f190d",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
    ],
  )