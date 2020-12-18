defmodule Exla.Aot.Compile do
  @moduledoc false

  alias Exla.Aot.Codegen

  # We need to put everything inside of the TF checkout (this needs to be a function)
  @tf_checkout_path "/home/sean/projects/exla/tmp/exla_cache/tf-8a1bb87da5b8dcec1d7f0bf8baa43565c095830e"

  alias Exla.Computation

  def compile(computations, functions) do
    :ok =
      computations
      |> Enum.zip(functions)
      |> Enum.reduce(:ok, fn {comp, func}, _acc -> compile_function(comp, func) end)

    {:ok, nif_path} = write_nif_source_file(functions, "AotTest", "exla_aot_class", "nif")
    {:ok, build_path} = write_bazel_build_file("nif", functions)
    cwd = File.cwd!()
    System.cmd("bazel", ["build", "//tensorflow/compiler/xla/exla/aot:libnif.so"], cd: @tf_checkout_path)
    System.cmd("mv", ["bazel-bin/tensorflow/compiler/xla/exla/aot/libnif.so", cwd], cd: @tf_checkout_path)

    # clean_srcs(nif_path, build_path)
  end

  defp compile_function(%Computation{ref: comp}, {name, arity, args, _result_size}) do
    {:ok, pbtext_path} = write_graph_config_file({name, arity, args})
    Exla.NIF.compile_aot(comp, pbtext_path, get_aot_path(), "#{name}_#{arity}", "exla_aot_class")
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

  defp clean_srcs(nif_path, build_path) do
    File.rm!(nif_path)
    File.rm!(build_path)
  end

  defp get_aot_path, do: @tf_checkout_path <> "/tensorflow/compiler/xla/exla/aot/"

end