defmodule Nx.Defn.AOTTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  defmodule AOT do
    @behaviour Nx.Defn.Compiler

    def __stream__(_, _, _, _, _, _), do: raise("not implemented")
    def __jit__(_, _, _, _), do: raise("not implemented")

    def __aot__(dir, module, functions, _) do
      path = Path.join(dir, "#{module}.so")
      File.write!(path, "BINARY")
      results = Enum.map(functions, fn {_, fun, vars, _} -> fun.(vars) end)
      {:ok, results, path}
    end
  end

  def aot_module(config, functions, actual_arities \\ []) do
    tmp_dir = config.tmp_dir
    module = :"Elixir.Nx.Defn.AOTTest#{System.unique_integer([:positive])}"

    :ok = Nx.Defn.export_aot(tmp_dir, module, functions, compiler: AOT)

    functions =
      for {name, _, args, _} <- functions do
        aot_name = :"__aot_#{name}_#{length(args)}"
        arity = actual_arities[name] || length(args)
        args = Macro.generate_arguments(arity, __MODULE__)

        quote do
          defoverridable [{unquote(aot_name), unquote(arity)}]

          def unquote(aot_name)(unquote_splicing(args)) do
            Process.get(:aot).(unquote_splicing(args))
          end
        end
      end

    contents =
      quote do
        Nx.Defn.import_aot(unquote(tmp_dir), __MODULE__)

        # Overwrite the load function so we don't try to load the NIF
        defoverridable __on_load__: 0
        def __on_load__(), do: :ok

        # Now define a mock for each function we are going to call
        unquote(functions)
      end

    Module.create(module, contents, __ENV__)
    module
  end

  defp register_aot(fun) do
    Process.put(:aot, fun)
  end

  @tag :tmp_dir
  test "defines an AOT module", config do
    module =
      aot_module(config, [
        {:add, &Nx.add/2, [Nx.template({}, {:s, 64}), Nx.template({}, {:f, 32})], []}
      ])

    register_aot(fn <<1::64-native>>, <<2.0::32-float-native>> ->
      {:ok, [<<3.0::32-float-native>>]}
    end)

    assert module.add(Nx.tensor(1), Nx.tensor(2.0)) == Nx.tensor(3.0)
  end

  @tag :tmp_dir
  test "accepts numbers as templates and arguments", config do
    module = aot_module(config, [{:add, &Nx.add/2, [1, 2.0], []}])

    register_aot(fn <<1::64-native>>, <<2.0::32-float-native>> ->
      {:ok, [<<3.0::32-float-native>>]}
    end)

    assert module.add(1, 2.0) == Nx.tensor(3.0)
  end

  defn tuple({a, b}, c), do: {{a + c, b + c}, a - b}

  @tag :tmp_dir
  test "accepts tuples as inputs and outputs", config do
    module = aot_module(config, [{:tuple, &tuple/2, [{1, 2}, 3], []}], tuple: 3)

    register_aot(fn <<1::64-native>>, <<2::64-native>>, <<3::64-native>> ->
      {:ok, [<<4::64-native>>, <<5::64-native>>, <<-1::64-native>>]}
    end)

    assert module.tuple({1, 2}, 3) == {{Nx.tensor(4), Nx.tensor(5)}, Nx.tensor(-1)}
  end

  @tag :tmp_dir
  test "validates type, shape, and names", config do
    module = aot_module(config, [{:add, &Nx.add/2, [1, 2.0], []}])

    assert_raise ArgumentError,
                 ~r/Nx AOT-compiled function expected a tensor of type, shape, and names/,
                 fn -> module.add(Nx.tensor([1, 2, 3]), 2.0) end

    assert_raise ArgumentError,
                 ~r/Nx AOT-compiled function expected a tensor of type, shape, and names/,
                 fn -> module.add(1.0, 2.0) end
  end
end
