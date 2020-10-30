# Exla

Elixir XLA Client.

## Building

You need [Bazel](https://docs.bazel.build/versions/master/install.html) and [CUDA](#) >= 10.1.

Running `iex -S mix` inside the root directory goes through the whole build process. You might see something like:

```
ERROR: <builtin>: Couldn't build file stable-status.txt: Failed to determine workspace status: Process exited with status 127
```

The build will continue and build the libraries successfully, but it will say that the build did not succeed. I think it's a bug with Bazel. Elixir Make is configured to ignore these errors and continue anyway.

You'll also see a ton of warnings from compiling XLA from the TensorFlow tree. The build took ~5 minutes running on 12 threads on my machine.

## Running

You can build up computations but not compile them yet.