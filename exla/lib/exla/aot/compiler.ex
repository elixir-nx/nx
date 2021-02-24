defmodule EXLA.AOT.Compiler do
  @moduledoc false

  alias EXLA.AOT.Codegen
  alias EXLA.Computation

  require Logger
  @tf_rev "6af836f407f546cf2f9ab3b5fcb7a8285bda5c96"

  def compile(module_name, functions, _options) do
    lib_name = "nif"
    :ok = mk_aot_dir()

    # Compile each function to header/object
    src_paths = Enum.map(functions, &compile_function/1)

    # Write out a NIF/BUILD file
    {:ok, nif_path} = write_nif_source_file(functions, module_name, lib_name)
    {:ok, build_path} = write_bazel_build_file("nif", functions)

    # Get cwd for output `.so` path
    cwd = File.cwd!()

    # Make an erts symlink
    erts_path =
      List.to_string(
        :lists.concat([:code.root_dir(), '/erts-', :erlang.system_info(:version), '/include'])
      )

    erts_link_path = get_aot_path() <> "erts"
    _ = File.rm(Path.join(cwd, "libnif.so"))

    with {_, 0} <- System.cmd("ln", ["-s", erts_path, erts_link_path]),
         {_, 0} <-
           System.cmd("bazel", ["build", "//tensorflow/compiler/xla/exla/aot:libnif.so"],
             cd: get_tf_checkout_path(),
             stderr_to_stdout: true,
             into: IO.stream(:stdio, :line)
           ),
         {_, 0} <-
           System.cmd("mv", ["bazel-bin/tensorflow/compiler/xla/exla/aot/libnif.so", cwd],
             cd: get_tf_checkout_path()
           ) do
      clean_srcs(nif_path, build_path, src_paths)
    else
      _ ->
        clean_srcs(nif_path, build_path, src_paths)
        Logger.error("Unable to complete AOT compilation, something went wrong.")
    end
  end

  defp compile_function({%Computation{ref: comp}, name, arity, args, sizes}) do
    target_triple = get_target_triple()
    {:ok, pbtext_path} = write_graph_config_file(name, arity, args, Enum.count(sizes))

    src_paths =
      EXLA.NIF.compile_aot(
        comp,
        pbtext_path,
        get_aot_path(),
        "#{name}_#{arity}",
        "#{name}_#{arity}_class",
        target_triple
      )
      |> unwrap!()

    [pbtext_path | Tuple.to_list(src_paths)]
  end

  defp get_target_triple() do
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

  defp mk_aot_dir() do
    aot_path = get_aot_path()
    File.rm_rf!(aot_path)

    case File.mkdir(aot_path) do
      :ok -> :ok
      {:error, :eexist} -> :ok
      err -> err
    end
  end

  defp write_graph_config_file(name, arity, args, num_results) do
    pbtext = Codegen.generate_graph_config_file(args, num_results)
    pbtext_path = get_aot_path() <> "#{name}_#{arity}.pbtxt"
    {:ok, file} = File.open(pbtext_path, [:write])
    :ok = IO.binwrite(file, pbtext)
    {File.close(file), pbtext_path}
  end

  defp write_nif_source_file(functions, target_module, nif_name) do
    src = Codegen.generate_nif_source_file(functions, target_module)
    nif_path = get_aot_path() <> nif_name <> ".cc"
    {:ok, file} = File.open(nif_path, [:write])
    :ok = IO.binwrite(file, src)
    {File.close(file), nif_path}
  end

  defp write_bazel_build_file(nif_name, functions) do
    build = Codegen.generate_bazel_build_file(nif_name, functions)
    build_path = get_aot_path() <> "BUILD"
    {:ok, file} = File.open(build_path, [:write])
    :ok = IO.binwrite(file, build)
    {File.close(file), build_path}
  end

  defp clean_srcs(nif_path, build_path, src_paths) do
    File.rm(nif_path)
    File.rm(build_path)
    File.rm(get_aot_path() <> "erts")

    src_paths
    |> Enum.reduce(:ok, fn path, _ -> File.rm(path) end)
  end

  defp get_aot_path, do: get_tf_checkout_path() <> "/tensorflow/compiler/xla/exla/aot/"

  defp get_tf_checkout_path do
    Path.join([
      get_exla_cache(),
      "tf-#{@tf_rev}",
      "erts-#{:erlang.system_info(:version)}"
    ])
  end

  defp get_exla_cache,
    do:
      System.get_env("EXLA_CACHE") ||
        Path.join(
          System.get_env("TEMP") || Path.join(System.get_env("HOME"), ".cache"),
          "exla"
        )

  defp unwrap!({:ok, return}), do: return
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end