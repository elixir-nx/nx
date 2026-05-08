defmodule EXLA.CustomCallAliasTest do
  use EXLA.Case, async: false

  import Nx.Defn

  alias EXLA.Test.QRAliasBlock

  defmodule BuiltinFun do
    import Nx.Defn

    defn qr(t), do: Nx.LinAlg.qr(t)
    defn eigh(t), do: Nx.LinAlg.eigh(t)
  end

  defmodule Fun do
    import Nx.Defn

    alias EXLA.Test.QRAliasBlock

    defn qr_alias_fn(t) do
      q_out = Nx.template({3, 3}, {:f, 32})
      r_out = Nx.template({3, 4}, {:f, 32})

      Nx.block(%QRAliasBlock{}, [t], {q_out, r_out}, fn _, t2 ->
        Nx.LinAlg.qr(t2, mode: :reduced)
      end)
    end
  end

  @plugin_relative ~c"test/exla_qr_alias.so"

  defp plugin_path do
    :filename.join(:code.priv_dir(:exla), @plugin_relative)
  end

  defp mlir_via_jit_apply!(fun, args) when is_function(fun) and is_list(args) do
    try do
      Nx.Defn.jit_apply(fun, args,
        compiler: EXLA,
        module_compilation: :to_mlir
      )
    catch
      :throw, {:mlir_module, ref, used_inputs, output_container} ->
        %{
          mlir_module: EXLA.MLIR.Module.as_string(%EXLA.MLIR.Module{ref: ref}),
          used_inputs: used_inputs,
          output_container: output_container
        }
    end
  end

  defp load_plugin! do
    path = List.to_string(plugin_path())

    unless File.exists?(path) do
      flunk("""
      Missing #{path}. Build EXLA with MIX_ENV=test so the alias dylib is compiled \
      (see Makefile target exla_qr_alias.so).
      """)
    end

    case EXLA.NIF.load_dylib(path) do
      :ok ->
        :ok

      other ->
        flunk("load_dylib(#{path}) expected :ok, got: #{inspect(other)}")
    end
  end

  test "builtin QR lowering includes qr_cpu_custom_call_f32 in MLIR" do
    arg = Nx.iota({3, 4}, type: {:f, 32})
    assert %{mlir_module: mlir} = mlir_via_jit_apply!(&BuiltinFun.qr/1, [arg])

    assert mlir =~ "@qr_cpu_custom_call_f32("
    refute mlir =~ "qr_cpu_custom_call_f32_exla_alias"
  end

  test "builtin integer QR lowering converts operand and uses f32 target in MLIR" do
    arg = Nx.iota({3, 4}, type: {:s, 32})
    assert %{mlir_module: mlir} = mlir_via_jit_apply!(&BuiltinFun.qr/1, [arg])

    assert mlir =~ "stablehlo.convert"
    assert mlir =~ "@qr_cpu_custom_call_f32("
    refute mlir =~ "@qr_cpu_custom_call_s32("
    refute mlir =~ "qr_cpu_custom_call_f32_exla_alias"
  end

  test "builtin Eigh lowering includes eigh_cpu_custom_call_f32 in MLIR" do
    arg = Nx.iota({3, 3}, type: {:f, 32})
    assert %{mlir_module: mlir} = mlir_via_jit_apply!(&BuiltinFun.eigh/1, [arg])

    refute mlir =~ "stablehlo.convert"
    assert mlir =~ "@eigh_cpu_custom_call_f32("
  end

  test "builtin integer Eigh lowering converts operand and uses f32 target in MLIR" do
    arg = Nx.iota({3, 3}, type: {:s, 32})
    assert %{mlir_module: mlir} = mlir_via_jit_apply!(&BuiltinFun.eigh/1, [arg])

    assert mlir =~ "stablehlo.convert"
    assert mlir =~ "@eigh_cpu_custom_call_f32("
    refute mlir =~ "@eigh_cpu_custom_call_s32("
  end

  test "QR alias plugin: MLIR uses alias name and not the builtin target string" do
    load_plugin!()

    arg = Nx.iota({3, 4}, type: {:f, 32})
    assert %{mlir_module: mlir} = mlir_via_jit_apply!(&Fun.qr_alias_fn/1, [arg])

    assert mlir =~ "qr_cpu_custom_call_f32_exla_alias"
    refute mlir =~ "@qr_cpu_custom_call_f32("
  end

  test "QR alias plugin: JIT result matches builtin QR" do
    load_plugin!()

    t = Nx.iota({3, 4}, type: {:f, 32})
    exp = EXLA.jit(fn t -> Nx.LinAlg.qr(t) end).(t)
    act = EXLA.jit(&Fun.qr_alias_fn/1).(t)

    assert Nx.all_close(elem(exp, 0), elem(act, 0), atol: 1.0e-4, rtol: 1.0e-4)
    assert Nx.all_close(elem(exp, 1), elem(act, 1), atol: 1.0e-4, rtol: 1.0e-4)
  end
end
