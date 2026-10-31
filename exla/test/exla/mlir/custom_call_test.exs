defmodule EXLA.MLIR.CustomCallTest do
  use EXLA.Case, async: true

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Module
  alias EXLA.MLIR.Value

  # The CPU custom call is more precise than the generic QR used on other platforms
  # f16 machine epsilon is ~9.77e-4; atol must be above that
  defp tol_opts({:f, 16}, _platform), do: [atol: 7.0e-3, rtol: 7.0e-3]
  defp tol_opts({:bf, 16}, :host), do: [atol: 6.0e-2, rtol: 6.0e-2]
  defp tol_opts({:bf, 16}, _platform), do: [atol: 2.5e-1, rtol: 2.5e-1]
  defp tol_opts({:f, 64}, _platform), do: [atol: 1.0e-14, rtol: 1.0e-15]
  defp tol_opts({:f, 32}, :host), do: [atol: 1.0e-6, rtol: 1.0e-6]
  defp tol_opts({:f, 32}, _platform), do: [atol: 1.0e-5, rtol: 1.0e-5]

  describe "qr" do
    for type <- [bf: 16, f: 16, f: 32, f: 64] do
      test "works for input type #{inspect(type)}" do
        tol_opts = tol_opts(unquote(type), EXLAHelpers.client().platform)

        square = Nx.iota({4, 4}, type: unquote(type))
        tall = Nx.iota({4, 3}, type: unquote(type))
        wide = Nx.iota({3, 4}, type: unquote(type))

        fun =
          EXLA.jit(fn t ->
            {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
            Nx.dot(q, r)
          end)

        assert_all_close(fun.(square), square, tol_opts)
        assert_all_close(fun.(tall), tall, tol_opts)
        assert_all_close(fun.(wide), wide, tol_opts)
      end
    end
  end

  test "MLIR attribute names cannot override or duplicate attributes" do
    typespec = EXLA.Typespec.tensor({:f, 32}, {1})

    for mlir_attributes <- [
          [{"call_target_name", ~s("replacement")}],
          [{"test.attribute", "1 : i64"}, {"test.attribute", "2 : i64"}]
        ] do
      assert_raise ArgumentError, ~r/must be unique and cannot override/, fn ->
        Module.new([typespec], [typespec], fn function ->
          [argument] = Function.get_arguments(function)
          Value.custom_call([argument], [typespec], "original", [], mlir_attributes)
        end)
      end
    end
  end

  test "string MLIR attribute names preserve backend config without creating atoms" do
    typespec = EXLA.Typespec.tensor({:f, 32}, {1})
    name = "test.attribute_#{System.unique_integer([:positive])}"

    assert_raise ArgumentError, fn -> String.to_existing_atom(name) end

    mlir =
      Module.new([typespec], [typespec], fn function ->
        [argument] = Function.get_arguments(function)

        [result] =
          Value.custom_call(
            [argument],
            [typespec],
            "original",
            [{"test_backend", "42 : i64"}],
            [{name, "1 : i64"}]
          )

        Value.func_return(function, [result])
        Module.as_string(function.module)
      end)

    assert mlir =~ "backend_config = {test_backend = 42 : i64}"
    assert mlir =~ "#{name} = 1 : i64"
    assert_raise ArgumentError, fn -> String.to_existing_atom(name) end
  end

  test "a binary backend config is an escaped string of the given api_version" do
    typespec = EXLA.Typespec.tensor({:f, 32}, {1})

    mlir =
      Module.new([typespec], [typespec], fn function ->
        [argument] = Function.get_arguments(function)

        [result] =
          Value.custom_call([argument], [typespec], "legacy", ~s({"k": "v"}), [], 2)

        Value.func_return(function, [result])
        Module.as_string(function.module)
      end)

    assert mlir =~ "api_version = 2 : i32"
    assert mlir =~ ~S(backend_config = "{\22k\22: \22v\22}")
  end

  test "extra results are dropped from the block outputs" do
    t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0], [1.0, 0.0, 1.0]])

    fun =
      EXLA.jit(fn t ->
        Nx.block(%EXLA.Test.QRQBlock{}, [t], Nx.template({4, 3}, :f32), fn _, t ->
          t |> Nx.LinAlg.qr(mode: :reduced) |> elem(0)
        end)
      end)

    {q, _r} = Nx.LinAlg.qr(Nx.backend_copy(t, Nx.BinaryBackend), mode: :reduced)
    assert_all_close(Nx.abs(fun.(t)), Nx.abs(q), atol: 1.0e-5, rtol: 1.0e-5)
  end
end
