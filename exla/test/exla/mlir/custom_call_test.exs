defmodule EXLA.MLIR.CustomCallTest do
  use EXLA.Case, async: true

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Module
  alias EXLA.MLIR.Value

  describe "qr" do
    for type <- [bf: 16, f: 16, f: 32, f: 64] do
      tol_opts =
        case type do
          {:f, 16} ->
            # f16 machine epsilon is ~9.77e-4; atol must be above that
            [atol: 7.0e-3, rtol: 7.0e-3]

          {:bf, 16} ->
            [atol: 6.0e-2, rtol: 6.0e-2]

          {:f, 64} ->
            [atol: 1.0e-14, rtol: 1.0e-15]

          {:f, 32} ->
            [atol: 1.0e-6, rtol: 1.0e-6]
        end

      test "works for input type #{inspect(type)}" do
        square = Nx.iota({4, 4}, type: unquote(type))
        tall = Nx.iota({4, 3}, type: unquote(type))
        wide = Nx.iota({3, 4}, type: unquote(type))

        fun =
          EXLA.jit(fn t ->
            {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
            Nx.dot(q, r)
          end)

        assert_all_close(fun.(square), square, unquote(tol_opts))
        assert_all_close(fun.(tall), tall, unquote(tol_opts))
        assert_all_close(fun.(wide), wide, unquote(tol_opts))
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
end
