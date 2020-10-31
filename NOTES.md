# XLA Notes

Just my collection of notes on the internals of XLA.

## How it works

The basic workflow is:

1. Queue operations using `xla::XlaBuilder` and `xla::XlaOp` functions.
2. Build a computation from the queued operations using `xla::XlaBuilder::Build()`
3. Compile to an `xla::Executable` using an instance of `xla::Client` (`xla::LocalClient` for now)
4. Execute the `xla::Executable` using client instance `Compile` function

## File Structure

At a high-level, XLA consists of the following:

| Component    | File Path                                       | Short Description                                |
|--------------|-------------------------------------------------|--------------------------------------------------|
| `client`     | `tensorflow/tensorflow/compiler/xla/client`     | Public Client API for using the XLA Compiler     |
| `pjrt`       | `tensorflow/tensorflow/compiler/xla/pjrt`       | "Pretty much Just another RunTime", wraps Python?|
| `python`     | `tensorflow/tensorflow/compiler/xla/python`     | Python XLA Client                                |
| `python_api` | `tensorflow/tensorflow/compiler/xla/python_api` | API Exposed from Tensorflow API                  |
| `rpc`        | `tensorflow/tensorflow/compiler/xla/rpc`        | gRPC XLA Service implementation                  |
| `service`    | `tensorflow/tensorflow/compiler/xla/service`    | XLA Services for GPU, CPU, etc.                  |

TensorFlow uses Bazel, which gives you control over which resources can be brought in as external dependencies. The following modules from the base XLA directory are available for public use:

| Name                     | Bazel Dependency Path                                     | Type            | Header/Source File                                   |
|--------------------------|-----------------------------------------------------------|-----------------|------------------------------------------------------|
| `xla_data_proto`         | `@org_tensorflow//tensorflow/compiler/xla:xla_data_proto` | Protocol Buffer | `tensorflow/tensorflow/compiler/xla/xla_data.proto`  |
| `xla_proto`              | `@org_tensorflow//tensorflow/compiler/xla:xla_proto`      | Protocol Buffer | `tensorflow/tensorflow/compiler/xla/xla.proto`       |
| `status_macros`          | `@org_tensorflow//tensorflow/compiler/xla:status_macros`  | Library         | `tensorflow/tensorflow/compiler/xla/status_macros.h` |
| `status`                 | `@org_tensorflow//tensorflow/compiler/xla:status`         | Library         | `tensorflow/tensorflow/compiler/xla/status.h`        |
| `statusor`               | `@org_tensorflow//tensorflow/compiler/xla:statusor`       | Library         | `tensorflow/tensorflow/compiler/xla/statusor.h`      |
| `util`                   | `@org_tensorflow//tensorflow/compiler/xla:util`           | Library         | `tensorflow/tensorflow/compiler/xla/util.h`          |
| `protobuf_util`          | `@org_tensorflow//tensorflow/compiler/xla:protobuf_util`  | Library         | `tensorflow/tensorflow/compiler/xla/protobuf_util.h` |
| `shape_util`             |                                                           |                 |                                                      |
| `literal`                |                                                           |                 |                                                      |
| `literal_util`           |                                                           |                 |                                                      |
| `metric_table_report`    |                                                           |                 |                                                      |
| `device_util`            |                                                           |                 |                                                      |
| `array2d`                |                                                           |                 |                                                      |
| `executable_run_options` |                                                           |                 |                                                      |
| `shape_tree`             |                                                           |                 |                                                      |
| `shape_layout`           |                                                           |                 |                                                      |
| `window_util`            |                                                           |                 |                                                      |
| `reference_util`         |                                                           |                 |                                                      |

## Client

Most of the important parts of the public API are in `client`. `xla::Client` provides an interface for working with services on the backend. Clients can either be local or remote. Local Clients are an instance of `xla::LocalClient` and are attached to devices on the host machine. Remote clients are an instance of `xla::Client` with an attached `GRPCService`.

For now, we just work with local clients.