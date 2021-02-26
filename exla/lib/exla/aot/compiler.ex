defmodule EXLA.AOT.Compiler do
  @moduledoc false

  alias EXLA.AOT.Codegen
  alias EXLA.Computation

  require Logger
  @tf_rev "6af836f407f546cf2f9ab3b5fcb7a8285bda5c96"

  def compile(module_name, functions, _options) do
    lib_name = "libnif"
    aot_dir = "aot#{System.unique_integer([:positive])}"
    aot_relative_path = "tensorflow/compiler/xla/exla/#{aot_dir}"
    tf_path = tf_checkout_path()
    target_triple = target_triple()
    target_path = "#{module_name}.#{nif_extension(target_triple)}"

    config = %{
      aot_dir: aot_dir,
      aot_path: Path.join(tf_path, aot_relative_path),
      aot_relative_path: aot_relative_path,
      lib_name: lib_name,
      module_name: module_name,
      target_triple: target_triple(),
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

    case System.cmd("bazel", ["build", "//#{config.aot_relative_path}:#{config.lib_name}.so"],
             cd: config.tf_path,
             stderr_to_stdout: true,
             into: IO.stream(:stdio, :line)
           ) do
      {_, 0} ->
        so =
          config.tf_path
          |> Path.join("bazel-bin/#{config.aot_relative_path}/#{config.lib_name}.so")
          |> File.read!()

        File.write!(config.target_path, so)
        :ok

      {_, _} ->
        Logger.error("Unable to complete AOT compilation, something went wrong.")
        :error
    end
  end

  defp compile_function({%Computation{ref: comp, output_shape: out_shape}, name, args}, config) do
    %EXLA.Shape{dtype: {:t, shapes}} = out_shape
    arity = length(args)

    sizes =
      Enum.map(shapes, fn shape ->
        {_, size} = shape.dtype
        Nx.size(shape.dims) * div(size, 8)
      end)

    {:ok, pbtext_path} = write_graph_config_file(name, arity, args, sizes, config)

    header_path = Path.join(config.aot_path, "#{name}_#{arity}.h")
    object_path = Path.join(config.aot_path, "#{name}_#{arity}.o")

    EXLA.NIF.compile_aot(
      comp,
      pbtext_path,
      header_path,
      object_path,
      "#{name}_#{arity}",
      "#{name}_#{arity}_class",
      config.target_triple
    )
    |> unwrap!()

    {name, arity, args, sizes}
  end

  defp write_graph_config_file(name, arity, args, sizes, config) do
    pbtext = Codegen.generate_graph_config_file(args, sizes)
    pbtext_path = Path.join(config.aot_path, "#{name}_#{arity}.pbtxt")
    File.write!(pbtext_path, pbtext)
    {:ok, pbtext_path}
  end

  defp write_nif_source_file(functions, config) do
    src = Codegen.generate_nif_source_file(functions, config.module_name, config.aot_relative_path)
    nif_path = Path.join(config.aot_path, config.lib_name <> ".cc")
    File.write!(nif_path, src)
    :ok
  end

  defp write_bazel_build_file(functions, config) do
    build = Codegen.generate_bazel_build_file(functions, config.lib_name)
    build_path = Path.join(config.aot_path, "BUILD")
    File.write!(build_path, build)
    :ok
  end

  defp nif_extension(target_triple) do
    if String.ends_with?(target_triple, "-windows"), do: :dll, else: :so
  end

  defp target_triple() do
    # See supported triples: https://github.com/tensorflow/tensorflow/blob/e687cab61615a95b8aea79120c9f168d4cc30955/tensorflow/compiler/aot/tfcompile.bzl
    case :os.type() do
      {:unix, :linux} ->
        "x86_64-pc-linux"

      {:win32, _} ->
        "x86_64-none-windows"

      {:unix, osname} ->
        arch_str = :erlang.system_info(:system_architecture)
        [arch, vendor | _] = arch_str |> List.to_string() |> String.split("-")
        arch <> "-" <> vendor <> "-" <> Atom.to_string(osname)
    end
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

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end