defmodule EXLA.MLIR.ExecutableTest do
  use EXLA.Case, async: true

  test "mvp" do
    # TO-DO (mlir): this will probably be reorganized in the end
    # This test is being added as an MVP for MLIR compilation

    t1 = Nx.broadcast(0.0, {2, 3, 1})
    t2 = Nx.broadcast(1.0, {2, 3, 1})

    result =
      EXLA.jit_apply(
        fn t1, t2 ->
          t1
          |> Nx.add(t2)
          |> then(&{Nx.add(t2, &1), Nx.subtract(t2, &1)})
          |> then(&elem(&1, 0))
        end,
        [t1, t2],
        compiler_mode: :mlir
      )

    # expected = {Nx.add(t2, t2), t1}
    expected = Nx.add(t2, t2)
    assert_equal(result, expected)
  end
end
