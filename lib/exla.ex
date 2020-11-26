defmodule Exla do
  @moduledoc """
  Bindings and Nx integration for [Google's XLA](https://www.tensorflow.org/xla/).

  ## defn compiler

  Most often, this library will be used as `Nx.Defn` compiler, like this:

      @defn_compiler Exla
      defn softmax(tensor) do
        Nx.exp(n) / Nx.sum(Nx.exp(n))
      end

  Then, every time `softmax/1` is called, Exla will just-in-time (JIT)
  compile a native implementation of the function above, tailored for the
  type and shape of the given tensor. Ahead-of-time (AOT) compilation is
  planned for future versions.

  Exla is able to compile to the CPU or GPU, by customizing the default
  client or specifying your own client:

      @defn_compiler {Exla, client: :cuda}
      defn softmax(tensor) do
        Nx.exp(n) / Nx.sum(Nx.exp(n))
      end

  Read the "Client" section below for more information.

  ### Options

  The options accepted by the Exla compiler are:

    * `:client` - an atom representing the client to use. Defaults
      to `:default`.

  ## Clients

  The `Exla` library uses a client for compiling and executing code.
  Those clients are typically bound to a platform, such as CPU or
  GPU.

  Those clients are singleton resources on Google's XLA library,
  therefore they are treated as a singleton resource on this library
  too. You can configure a client via the application environment.
  For example, to configure the default client:

      config :exla, :clients,
        default: [platform: :host]

  `platform: :host` is the default value. You can configure it to
  use the GPU though:

      config :exla, :clients,
        default: [platform: :cuda]

  You can also specify multiple clients for different platforms:

      config :exla, :clients,
        default: [platform: :host],
        cuda: [platform: :cuda]

  While specifying multiple clients is possible, keep in mind you
  want a single client per platform. If you have multiple clients
  per platform, they can race each other and fight for resources,
  such as memory. Therefore, we recommend developers to use the
  `:default` client as much as possible.

  ## Docker considerations

  Exla should run fine on Docker with one important consideration:
  you must not start the Erlang VM as the root process in Docker.
  That's because when the Erlang VM runs as root, it has to manage
  all child programs.

  At the same time, Google XLA's shells out to child program during
  compilation and it must retain control over how child programs
  terminate.

  To address this, simply make sure you wrap the Erlang VM in
  another process, such as the shell one. In other words, if you
  are using releases, instead of this:

      RUN path/to/release start

  do this:

      RUN sh -c "path/to/release start"

  If you are Mix inside your Docker containers, instead of this:

      RUN mix run

  do this:

      RUN sh -c "mix run"

  """

  @behaviour Nx.Defn.Compiler
  @impl true
  defdelegate __compile__(kind, meta, name, args, ast, opts), to: Exla.Defn
end
