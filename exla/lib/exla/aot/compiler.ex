defmodule EXLA.AOT.Compiler do
  @moduledoc false

  alias EXLA.AOT.Codegen
  alias EXLA.Computation

  require Logger

  # We need to put everything inside of the TF checkout
  @tf_checkout_path "/tf-6af836f407f546cf2f9ab3b5fcb7a8285bda5c96"

  def compile(computations, functions, module_name, lib_name \\ "nif") do
    :ok = mk_aot_dir()

    # Compile each function to header/object
    src_paths =
      computations
      |> Enum.zip(functions)
      |> Enum.flat_map(fn {comp, func} -> compile_function(comp, func) end)

    # Write out a NIF/BUILD file
    {:ok, nif_path} = write_nif_source_file(functions, module_name, "exla_aot_class", lib_name)
    {:ok, build_path} = write_bazel_build_file("nif", functions)

    # Get cwd for output `.so` path
    cwd = File.cwd!()

    # Make an erts symlink
    erts_path =
      List.to_string(
        :lists.concat([:code.root_dir(), '/erts-', :erlang.system_info(:version), '/include'])
      )

    erts_link_path = get_aot_path() <> "erts"

    with {_, 0} <- System.cmd("ln", ["-s", erts_path, erts_link_path]),
         {_, 0} <-
           System.cmd("bazel", ["build", "//tensorflow/compiler/xla/exla/aot:libnif.so"],
             cd: get_tf_checkout_path()
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

  defp compile_function(%Computation{ref: comp}, {name, arity, args, _result_size}) do
    {:ok, pbtext_path} = write_graph_config_file({name, arity, args})

    src_paths =
      EXLA.NIF.compile_aot(
        comp,
        pbtext_path,
        get_aot_path(),
        "#{name}_#{arity}",
        "exla_aot_class"
      )
      |> unwrap!()

    [pbtext_path | Tuple.to_list(src_paths)]
  end

  defp mk_aot_dir() do
    aot_path = get_aot_path()

    case File.mkdir(aot_path) do
      :ok -> :ok
      {:error, :eexist} -> :ok
      err -> err
    end
  end

  defp write_graph_config_file({name, arity, args}) do
    pbtext = Codegen.generate_graph_config_file(args)
    pbtext_path = get_aot_path() <> "#{name}_#{arity}.pbtxt"
    {:ok, file} = File.open(pbtext_path, [:write])
    :ok = IO.binwrite(file, pbtext)
    {File.close(file), pbtext_path}
  end

  defp write_nif_source_file(functions, target_module, class_name, nif_name) do
    src = Codegen.generate_nif_source_file(functions, target_module, class_name)
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
  defp get_tf_checkout_path, do: get_exla_cache() <> @tf_checkout_path
  defp get_exla_cache, do: System.get_env("EXLA_CACHE")

  defp unwrap!({:ok, return}), do: return
end
