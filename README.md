# Exla

Elixir XLA Client for compiling and running Elixir code on CPU/GPU/TPU.

## Building

The easiest way to build is with [Docker](https://docs.docker.com/get-docker/). You'll also need [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

To build, clone this repo and run:

```
$ docker build --rm -t exla:cuda10.1 .
```

Then to run:

```
$ docker run -it \
    -v $PWD:$PWD \
    -w $PWD \
    --rm exla:cuda10.1 bash
```

Inside the container you can interact with the API from IEX using:

```
exla/#: iex -S mix
```

Or you can run an example:

```
exla/#: mix run examples/basic_addition.exs
```