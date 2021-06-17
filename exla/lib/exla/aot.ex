defmodule EXLA.AOT do
  @moduledoc """
  High-level functionality for AOT compilation.
  """

  alias EXLA.AOT.Codegen
  alias EXLA.Computation

  require Logger
  @tf_rev "6af836f407f546cf2f9ab3b5fcb7a8285bda5c96"

  @doc """
  Compiles a shared object file at the given `output_dir`
  for `module_name` with the given `functions`.

  The shared object file will be called "libnif.so"
  (or "libnif.dll" for Windows) and be placed on the given
  `output_dir`. This function returns `{:ok, path_to_nif}`
  or `{:error, Exception.t}`.

  `module_name` is an atom representing a module. `functions`
  is a list of 4-element tuples  of shape
  `{computation, name, arity, shapes}` where:

    * `computation` is an EXLA computation which returns
      a tuple

    * `name` is the name of the function to be defined as NIF

    * `arity` is the arity of the function to be defined as NIF

    * `shapes` is a list of 2-element tuples with the computation
      arguments. The first element is the zero-based index of the
      argument and the second element is its EXLA.Shape

  Each compiled NIF expects binaries as arguments, as described
  by the respective shapes, and it returns either `{:ok, list}`
  or `{:error, charlist}`, where `list` is a list of binaries
  representing each element of the output tuple.

  ## Options

  This function accepts the following options:

    * `:runtimes` - some features of the computation
      graph, such as dot (`matmul`) and conv (`conv2d`)
      require specific runtimes to operate. You need to
      explicitly pass said runtimes as argument:

          runtimes: [:matmul, :conv2d]

      The list of runtimes can be found on Tensorflow
      source [in this directory](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/service/cpu),
      starting with the `runtime_` prefix

    * `:bazel_flags` - flags that customize `bazel build` command

    * `:bazel_env` - flags that customize `bazel build` environment.
      It must be a list of tuples where the env key and env value
      are binaries

  Also see the options in `EXLA.Compilation.compile_aot/7`.
  """
  def compile(output_dir, module_name, functions, options \\ [])
      when is_binary(output_dir) and is_atom(module_name) and is_list(functions) and
             is_list(options) do
    lib_name = "libnif"
    aot_dir = "aot#{System.unique_integer([:positive])}"
    aot_relative_path = "tensorflow/compiler/xla/exla/#{aot_dir}"
    tf_path = tf_checkout_path()
    target_triple = options[:target_triple] || Computation.target_triple()
    target_path = Path.join(output_dir, "#{module_name}.#{nif_extension(target_triple)}")

    config = %{
      aot_dir: aot_dir,
      aot_path: Path.join(tf_path, aot_relative_path),
      aot_relative_path: aot_relative_path,
      bazel_flags: options[:bazel_flags] || [],
      bazel_env: options[:bazel_env] || [],
      lib_name: lib_name,
      module_name: module_name,
      runtimes: options[:runtimes] || [],
      target_features: options[:target_features],
      target_triple: target_triple,
      target_path: target_path,
      tf_path: tf_path
    }

    try do
      compile(functions, config)
    after
      File.rm_rf!(config.aot_path)
    end
  end

  defp compile(functions, config) do
    File.rm_rf!(config.aot_path)
    File.mkdir_p!(config.aot_path)

    # Compile each function to header/object and return updated tuples
    functions = Enum.map(functions, &compile_function(&1, config))

    # Write out a NIF/BUILD file
    :ok = write_nif_source_file(functions, config)
    :ok = write_bazel_build_file(functions, config)

    # Make an erts symlink
    erts_path = Path.join([:code.root_dir(), "erts-#{:erlang.system_info(:version)}", "include"])
    erts_link_path = Path.join(config.aot_path, "erts")

    case File.ln_s(erts_path, erts_link_path) do
      :ok -> :ok
      {:error, _} -> File.cp_r!(erts_path, erts_link_path)
    end

    system_args =
      ["build"] ++ config.bazel_flags ++ ["//#{config.aot_relative_path}:#{config.lib_name}.so"]

    system_opts = [
      cd: config.tf_path,
      env: config.bazel_env,
      into: IO.stream(:stdio, :line),
      stderr_to_stdout: true
    ]

    case System.cmd("bazel", system_args, system_opts) do
      {_, 0} ->
        so =
          config.tf_path
          |> Path.join("bazel-bin/#{config.aot_relative_path}/#{config.lib_name}.so")
          |> File.read!()

        File.write!(config.target_path, so)
        {:ok, config.target_path}

      {_, _} ->
        message = "unable to complete AOT compilation, something went wrong"
        {:error, RuntimeError.exception(message)}
    end
  end

  defp compile_function({%Computation{output_shape: out_shape} = comp, name, arity, args}, config) do
    %EXLA.Shape{dtype: {:t, shapes}} = out_shape

    sizes =
      Enum.map(shapes, fn shape ->
        {_, size} = shape.dtype
        Nx.size(shape.dims) * div(size, 8)
      end)

    {:ok, pbtext_path} = write_graph_config_file(name, arity, args, sizes, config)

    header_path = Path.join(config.aot_path, "#{name}_#{arity}.h")
    object_path = Path.join(config.aot_path, "#{name}_#{arity}.o")

    :ok =
      Computation.compile_aot(
        comp,
        pbtext_path,
        header_path,
        object_path,
        "#{name}_#{arity}",
        "#{name}_#{arity}_class",
        target_triple: config.target_triple,
        target_features: config.target_features
      )

    {name, arity, args, sizes}
  end

  defp write_graph_config_file(name, arity, args, sizes, config) do
    pbtext = Codegen.generate_graph_config_file(args, sizes)
    pbtext_path = Path.join(config.aot_path, "#{name}_#{arity}.pbtxt")
    File.write!(pbtext_path, pbtext)
    {:ok, pbtext_path}
  end

  defp write_nif_source_file(functions, config) do
    src =
      Codegen.generate_nif_source_file(functions, config.module_name, config.aot_relative_path)

    nif_path = Path.join(config.aot_path, config.lib_name <> ".cc")
    File.write!(nif_path, src)
    :ok
  end

  defp write_bazel_build_file(functions, config) do
    build = Codegen.generate_bazel_build_file(functions, config.runtimes, config.lib_name)
    build_path = Path.join(config.aot_path, "BUILD")
    File.write!(build_path, build)
    :ok
  end

  defp nif_extension(target_triple) do
    if String.ends_with?(target_triple, "-windows"), do: :dll, else: :so
  end

  defp tf_checkout_path() do
    Path.join([
      exla_cache_dir(),
      "tf-#{@tf_rev}",
      "erts-#{:erlang.system_info(:version)}"
    ])
  end

  defp exla_cache_dir() do
    System.get_env("EXLA_CACHE") ||
      Path.join(
        System.get_env("TEMP") || Path.join(System.get_env("HOME"), ".cache"),
        "exla"
      )
  end
end
