defmodule EXLA.AOT.Codegen do
  @moduledoc false

  ## Bazel Attributes
  @bazel_deps_base "@org_tensorflow//tensorflow/compiler/"
  @bazel_deps [
    "tf2xla:xla_compiled_cpu_function",
    "xla:cpu_function_runtime",
    "xla:executable_run_options",
    "xla:types"
  ]
  @bazel_erts_glob "glob([\"erts/**/*.h\"], allow_empty=False)"

  ## Nif Attributes
  @erl_nif_path "tensorflow/compiler/xla/exla/aot/erts/erl_nif"

  ## Generating the graph config file

  def generate_graph_config_file(params) do
    feeds = build_feeds_str(params)
    fetches = build_fetch_str(params)
    feeds <> fetches
  end

  defp build_feeds_str(params) do
    params
    |> Enum.map(&build_feed_str/1)
    |> Enum.join("")
  end

  defp build_feed_str(%{id: _id, name: name, dims: dims}) do
    shape_str = build_shape_str(dims)

    """
    feed {
      id { node_name: #{str(name)} }
      #{shape_str}
    }
    """
  end

  defp build_shape_str(dims) do
    dims_str =
      dims
      |> Tuple.to_list()
      |> Enum.map(fn x -> "dim { size: #{Integer.to_string(x)} }" end)
      |> Enum.join("\n")

    """
    shape {
      #{dims_str}
    }
    """
  end

  defp build_fetch_str(params) do
    result_name =
      params
      |> Enum.map(& &1.id)
      |> Enum.join("_")
      |> Kernel.<>("_result")

    """
    fetch {
      id { node_name: #{str(result_name)} }
    }
    """
  end

  ## Generating the BUILD file

  def generate_bazel_build_file(fname, functions) do
    name = build_bazel_so_str(fname)
    srcs = build_bazel_srcs_str(fname, functions)
    deps = build_bazel_deps_str()
    linkopts = build_bazel_linkopts_str()
    build_bazel_file_str(name, srcs, deps, linkopts)
  end

  defp build_bazel_file_str(name, srcs, deps, linkopts) do
    """
    cc_binary(
      name = #{name},
      srcs = #{srcs},
      deps = #{deps},
      linkopts = #{linkopts},
      linkshared = 1,
    )
    """
  end

  defp build_bazel_so_str(fname) do
    str("lib" <> fname <> ".so")
  end

  defp build_bazel_deps_str do
    deps_str =
      @bazel_deps
      |> Enum.map(&str(@bazel_deps_base <> &1))
      |> Enum.join(", ")

    "[" <> deps_str <> "]"
  end

  defp build_bazel_srcs_str(fname, functions) do
    cc_name = str(fname <> ".cc")

    src_files =
      functions
      |> Enum.flat_map(fn {name, arity, _, _} ->
        [str(Atom.to_string(name) <> "_#{arity}.h"), str(Atom.to_string(name) <> "_#{arity}.o")]
      end)
      |> Enum.join(", ")

    "[" <> cc_name <> ", " <> src_files <> "]" <> "+" <> @bazel_erts_glob
  end

  defp build_bazel_linkopts_str do
    "[" <> str("-shared") <> "]"
  end

  ## Generating the NIF Source File

  def generate_nif_source_file(functions, target_module, class_name) do
    include_block_str = build_include_block(functions)
    error_block_str = build_error_helper_block()
    load_block_str = build_load_block()
    functions_str = build_nif_funcs(functions, class_name)
    nif_func_export_array_str = build_nif_func_export_array(functions)
    init_block_str = build_init_block(target_module)

    include_block_str <>
      error_block_str <>
      load_block_str <> functions_str <> nif_func_export_array_str <> init_block_str
  end

  defp build_include_block(functions) do
    function_includes =
      functions
      |> Enum.map(fn {name, arity, _, _} ->
        build_include_str(Atom.to_string(name) <> "_#{arity}")
      end)
      |> Enum.join("\n")

    function_includes <> "\n" <> build_include_str(@erl_nif_path) <> "\n"
  end

  defp build_include_str(path) do
    "#include " <> str(path <> ".h")
  end

  defp build_error_helper_block do
    """
    static ERL_NIF_TERM error(ErlNifEnv* env, const char* msg) {
      return enif_make_tuple2(env, enif_make_atom(env, \"error\"), enif_make_string(env, msg, ERL_NIF_LATIN1));
    }
    """
  end

  defp build_load_block do
    """
    static int load(ErlNifEnv* env, void **priv, ERL_NIF_TERM load_info) { return 0; }
    """
  end

  defp build_nif_funcs(functions, class_name) do
    functions
    |> Enum.map(&build_nif_func_block(&1, class_name))
    |> Enum.join("\n")
  end

  defp build_nif_func_block({name, arity, args, result_size} = function, class_name) do
    signature_str = build_nif_func_signature(function)

    args_str =
      args
      |> Enum.map(&build_nif_arg_retrieval_block({name, arity, args, result_size}, &1))
      |> Enum.join("\n")

    run_str = build_nif_run_block({name, arity, args, result_size})
    result_str = build_nif_results_block({name, arity, args, result_size}, 0)

    """
    #{signature_str}{
      #{class_name} #{name}_#{arity};
      #{args_str}
      #{run_str}
      #{result_str}
    }
    """
  end

  defp build_nif_func_signature({name, arity, _, _}) do
    """
    ERL_NIF_TERM #{Atom.to_string(name)}_#{arity}_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
    """
  end

  defp build_nif_arg_retrieval_block({name, arity, _, _}, %{id: id}) do
    error_msg = str("could not get argument #{id}")

    """
      ErlNifBinary arg#{id};
      if(!enif_inspect_binary(env, argv[#{id}], &arg#{id})) {
        return error(env, #{error_msg});
      }
      #{name}_#{arity}.set_arg#{id}_data(arg#{id}.data);
    """
  end

  defp build_nif_run_block({name, arity, _, _}) do
    """
    #{name}_#{arity}.Run();
    """
  end

  defp build_nif_results_block({name, arity, _, result_size}, result_num) do
    error_msg = str("could not get result #{result_num}")

    """
    ErlNifBinary result#{result_num};
    if(!enif_alloc_binary(#{result_size}, &result#{result_num})) {
      return error(env, #{error_msg});
    }
    unsigned char * result#{result_num}_bytes = reinterpret_cast<unsigned char *>(#{name}_#{arity}.result#{result_num}_data());
    std::memcpy(
      result#{result_num}.data,
      result#{result_num}_bytes,
      #{result_size}
    );
    return enif_make_tuple2(env, enif_make_atom(env, \"ok\"), enif_make_binary(env, &result#{result_num}));
    """
  end

  defp build_nif_func_export_array(functions) do
    functions_str =
      functions
      |> Enum.map(&build_nif_func_export/1)
      |> Enum.join(",\n")

    """
    static ErlNifFunc nif_funcs[] = {
      #{functions_str}
    };
    """
  end

  defp build_nif_func_export({name, arity, _, _}) do
    "{#{str(Atom.to_string(name))}, #{arity}, #{name}_#{arity}_nif}"
  end

  defp build_init_block(target_module) do
    """
    ERL_NIF_INIT(#{target_module}, nif_funcs, &load, NULL, NULL, NULL);
    """
  end

  ## Shared Helpers

  defp str(string), do: "\"" <> string <> "\""
end