defmodule EXLATest do
  use EXLA.Case, async: true

  doctest EXLA

  describe "integration" do
    @describetag :integration

    test "memory leak test" do
      test_data = Nx.broadcast(0.0, {1000, 1000}) |> Nx.backend_transfer(Nx.BinaryBackend)
      fun = EXLA.jit(fn x -> Nx.tan(x) end)

      for _ <- 1..2000 do
        fun.(test_data)
        Process.sleep(10)
        :erlang.garbage_collect()
      end
    end
  end

  defmodule ValidCompiler do
    def __jit__(key, vars, fun, args_list, opts) do
      __compile__(key, vars, fun, opts).(args_list)
    end

    def __compile__(_key, vars, fun, opts) do
      result = EXLA.to_mlir_module(fun, vars, Keyword.put(opts, :within_defn_compiler, true))
      throw({__MODULE__, result})
    end
  end

  defmodule InvalidCompiler do
    def __jit__(key, vars, fun, args_list, opts) do
      __compile__(key, vars, fun, opts).(args_list)
    end

    def __compile__(_key, vars, fun, opts) do
      # Keyword.delete to ensure default is false
      EXLA.to_mlir_module(fun, vars, Keyword.delete(opts, :within_defn_compiler))
    end
  end

  describe "to_mlir_module/3" do
    test "fails if the compiler doesn't set the nested compilation flag" do
      assert_raise BadArityError, fn ->
        Nx.Defn.jit_apply(&Nx.add/2, [1, 2], compiler: __MODULE__.InvalidCompiler)
      end
    end

    test "works if the compiler sets the nested compilation flag" do
      try do
        Nx.Defn.jit_apply(&Nx.add/2, [1, 2], compiler: __MODULE__.ValidCompiler)
      catch
        {__MODULE__.ValidCompiler, result} ->
          assert %{mlir_module: module, output_container: container, used_inputs: used_inputs} =
                   result

          assert module == """
                 module {
                   func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
                     %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
                     return %0 : tensor<i32>
                   }
                 }
                 """

          assert Nx.compatible?(container, Nx.template({}, :s32))

          assert MapSet.equal?(used_inputs, MapSet.new([0, 1]))
      end
    end
  end
end
