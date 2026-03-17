defmodule EXLA.MLIR.CustomCallTest do
  use EXLA.Case, async: true

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
end
