# Exla

Elixir XLA Client for compiling and running Elixir code on CPU/GPU/TPU.

## Building

You need [Bazel](https://docs.bazel.build/versions/master/install.html).

It's currently configured to build with GPU support by default. To disable this option and only target CPU, remove the `--config=cuda` line from `bazel build//exla:libexla.so --config=cuda` in `Makefile`.

Running `iex -S mix` inside the root directory goes through the whole build process. The build process takes a lot of time (~30 mins) and resources.

## GPU Support

You need CUDA 10.1 and GCC-8. Nothing else will work.

## Running

After running `iex -S mix`, you can build up computations using XLA Operations. Check out: [XLA Operation Semantics](https://www.tensorflow.org/xla/operation_semantics) for a list of supported operations.