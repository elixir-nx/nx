defmodule EXLA.AOT.Codegen do
  @moduledoc false

  @bazel_erts_glob "glob([\"erts/**/*.h\"], allow_empty=False)"
  @bazel_deps_base "@org_tensorflow//"
  @bazel_deps_runtime "tensorflow/compiler/xla/service/cpu:runtime_"

  @bazel_deps [
    "tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
    "tensorflow/compiler/xla:cpu_function_runtime",
    "tensorflow/compiler/xla:executable_run_options",
    "tensorflow/compiler/xla:types",
    "third_party/eigen3"
  ]

  ## Generating the graph config file

  def generate_graph_config_file(args, sizes) do
    feeds = build_feeds_str(args)
    fetches = build_fetch_str(sizes)
    feeds <> fetches
  end

  defp build_feeds_str(args) do
    args
    |> Enum.map(&build_feed_str/1)
    |> Enum.join("")
  end

  defp build_feed_str({i, %EXLA.Shape{dims: dims}}) do
    shape_str = build_shape_str(dims)

    """
    feed {
      id { node_name: "arg#{i}" }
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

  defp build_fetch_str(sizes) do
    result_ids =
      for i <- 0..(length(sizes) - 1) do
        """
        fetch {
          id { node_name: "result#{i}" }
        }
        """
      end

    result_ids |> Enum.join("\n")
  end

  ## Generating the BUILD file

  def generate_bazel_build_file(functions, runtimes, lib_name) do
    name = build_bazel_so_str(lib_name)
    srcs = build_bazel_srcs_str(functions, lib_name)
    deps = build_bazel_deps_str(runtimes)
    linkopts = build_bazel_linkopts_str()

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

  defp build_bazel_so_str(lib_name) do
    str(lib_name <> ".so")
  end

  defp build_bazel_deps_str(runtimes) do
    deps = @bazel_deps ++ Enum.map(runtimes, &"#{@bazel_deps_runtime}#{&1}")
    deps_str = Enum.map_join(deps, ", ", &str(@bazel_deps_base <> &1))
    "[" <> deps_str <> "]"
  end

  defp build_bazel_srcs_str(functions, lib_name) do
    cc_name = str(lib_name <> ".cc")

    src_files =
      functions
      |> Enum.flat_map(fn {name, arity, _, _} ->
        [str(Atom.to_string(name) <> "_#{arity}.h"), str(Atom.to_string(name) <> "_#{arity}.o")]
      end)
      |> Enum.join(", ")

    "[" <> cc_name <> ", " <> src_files <> "]" <> "+" <> @bazel_erts_glob
  end

  defp build_bazel_linkopts_str do
    "[" <> str("-shared") <> "," <> str("-lpthread") <> "]"
  end

  ## Generating the NIF Source File

  def generate_nif_source_file(functions, target_module, aot_relative_path) do
    define_block_str = build_define_block()
    include_block_str = build_include_block(functions, aot_relative_path)
    error_block_str = build_error_helper_block()
    load_block_str = build_load_block()
    functions_str = build_nif_funcs(functions)
    nif_func_export_array_str = build_nif_func_export_array(functions)
    init_block_str = build_init_block(target_module)

    define_block_str <>
      include_block_str <>
      error_block_str <>
      load_block_str <> functions_str <> nif_func_export_array_str <> init_block_str
  end

  defp build_define_block() do
    """
    #define EIGEN_USE_THREADS
    #define EIGEN_USE_CUSTOM_THREAD_POOL
    """
  end

  defp build_include_block(functions, aot_relative_path) do
    function_includes =
      functions
      |> Enum.map(fn {name, arity, _, _} ->
        build_include_str(Atom.to_string(name) <> "_#{arity}")
      end)
      |> Enum.join("\n")

    erl_nif_path = Path.join(aot_relative_path, "erts/erl_nif")
    function_includes <> "\n" <> build_include_str(erl_nif_path) <> "\n"
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

  defp build_nif_funcs(functions) do
    functions
    |> Enum.map(&build_nif_func_block(&1))
    |> Enum.join("\n")
  end

  defp build_nif_func_block({name, arity, args, result_sizes}) do
    class_name = "#{name}_#{arity}_class"
    signature_str = build_nif_func_signature(name, arity)

    args_str =
      args
      |> Enum.map(&build_nif_arg_retrieval_block(name, arity, &1))
      |> Enum.join("\n")

    run_str = build_nif_run_block(name, arity)
    result_str = build_nif_results_block(name, arity, result_sizes)

    """
    #{signature_str}{
      unsigned num_threads = std::thread::hardware_concurrency();
      Eigen::ThreadPool tp(num_threads);
      Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
      #{class_name} #{name}_#{arity};
      #{name}_#{arity}.set_thread_pool(&device);
      #{args_str}
      #{run_str}
      #{result_str}
    }
    """
  end

  defp build_nif_func_signature(name, arity) do
    """
    ERL_NIF_TERM #{Atom.to_string(name)}_#{arity}_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
    """
  end

  defp build_nif_arg_retrieval_block(name, arity, {i, _shape}) do
    error_msg = str("could not get argument at index #{i}")

    """
    ErlNifBinary arg#{i};
    if(!enif_inspect_binary(env, argv[#{i}], &arg#{i})) {
      return error(env, #{error_msg});
    }
    #{name}_#{arity}.set_arg#{i}_data(arg#{i}.data);
    """
  end

  defp build_nif_run_block(name, arity) do
    """
    #{name}_#{arity}.Run();
    """
  end

  defp build_nif_results_block(name, arity, result_sizes) do
    res_str =
      result_sizes
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {size, i} ->
        build_nif_single_result_block(name, arity, size, i)
      end)

    num_results = length(result_sizes)

    res_terms =
      result_sizes
      |> Enum.with_index()
      |> Enum.map_join(", ", fn {_, i} -> "result#{i}_term" end)

    """
    #{res_str}
    ERL_NIF_TERM result_tuple = enif_make_list(env, #{num_results}, #{res_terms});
    return enif_make_tuple2(env, enif_make_atom(env, \"ok\"), result_tuple);
    """
  end

  defp build_nif_single_result_block(name, arity, size, i) do
    error_msg = str("could not get result at index #{i}")

    """
    ErlNifBinary result#{i};
    if(!enif_alloc_binary(#{size}, &result#{i})) {
      return error(env, #{error_msg});
    }
    unsigned char * result#{i}_bytes = reinterpret_cast<unsigned char *>(#{name}_#{arity}.result#{i}_data());
    std::memcpy(
      result#{i}.data,
      result#{i}_bytes,
      #{size}
    );

    ERL_NIF_TERM result#{i}_term = enif_make_binary(env, &result#{i});
    """
  end

  defp build_nif_func_export_array(functions) do
    functions_str = Enum.map_join(functions, ",\n", &build_nif_func_export/1)

    """
    static ErlNifFunc nif_funcs[] = {
      #{functions_str}
    };
    """
  end

  defp build_nif_func_export({name, arity, _, _}) do
    "{#{str(Atom.to_string(name))}, #{arity}, #{name}_#{arity}_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND}"
  end

  defp build_init_block(target_module) do
    """
    ERL_NIF_INIT(#{target_module}, nif_funcs, &load, NULL, NULL, NULL);
    """
  end

  ## Shared Helpers

  defp str(string), do: "\"" <> string <> "\""
end
