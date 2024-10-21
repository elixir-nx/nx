defmodule EXLA.MLIR.CustomCallTest do
  use EXLA.Case, async: true

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  describe "custom_operator_lowering_functions" do
    test "maps a Nx.Defn.Expr op to it's custom lowering function" do
      fft_lowering_function = fn %Function{} = function, expr, args ->
        [op, opts] = args
        typespec = EXLA.Typespec.tensor(expr.type, expr.shape)

        args = [
          op,
          Value.constant(function, [opts[:length]], EXLA.Typespec.tensor({:s, 64}, {})),
          Value.constant(function, [opts[:axis]], EXLA.Typespec.tensor({:s, 64}, {})),
          Value.constant(function, [opts[:eps]], EXLA.Typespec.tensor({:f, 32}, {}))
        ]

        [result] =
          Value.call(
            function,
            args,
            %Function{name: "nx_iree_fft"},
            [typespec]
          )

        result
      end

      assert %{output_container: output_container, mlir_module: mlir_module} =
               EXLA.to_mlir_module(
                 fn a, b ->
                   x = Nx.add(a, b)
                   y = Nx.fft(x)
                   Nx.subtract(y, 10)
                 end,
                 [Nx.template({3, 4}, :f32), Nx.template({3, 4}, :f32)],
                 custom_operator_lowering_functions: %{fft: fft_lowering_function}
               )

      assert %Nx.Tensor{type: {:c, 64}, shape: {3, 4}} = output_container

      assert mlir_module == """
             "builtin.module"() ({
               "func.func"() <{function_type = (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>, sym_name = "main", sym_visibility = "public"}> ({
               ^bb0(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>):
                 %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
                 %1 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
                 %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                 %3 = "stablehlo.constant"() <{value = dense<1.000000e-10> : tensor<f32>}> : () -> tensor<f32>
                 %4 = "func.call"(%0, %1, %2, %3) <{callee = @nx_iree_fft}> : (tensor<3x4xf32>, tensor<i64>, tensor<i64>, tensor<f32>) -> tensor<3x4xcomplex<f32>>
                 %5 = "stablehlo.constant"() <{value = dense<10> : tensor<i32>}> : () -> tensor<i32>
                 %6 = "stablehlo.convert"(%5) : (tensor<i32>) -> tensor<complex<f32>>
                 %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<complex<f32>>) -> tensor<3x4xcomplex<f32>>
                 %8 = "stablehlo.subtract"(%4, %7) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
                 "func.return"(%8) : (tensor<3x4xcomplex<f32>>) -> ()
               }) : () -> ()
             }) : () -> ()
             """
    end
  end

  describe "qr" do
    for type <- [bf: 16, f: 16, f: 32, f: 64] do
      tol_opts =
        case type do
          {:f, 16} ->
            [atol: 1.0e-10, rtol: 1.0e-2]

          {:bf, 16} ->
            [atol: 1.0e-1, rtol: 1.0e-1]

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
end
