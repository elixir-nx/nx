defmodule EXLA.Defn.RecompilationWarningTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  setup do
    # Clear any recompilation counters from previous tests
    :ets.match_delete(EXLA.Defn.LockedCache, {{:counter, {:_, :_, :_}}, :_})
    :ok
  end

  test "warns for closure capturing different tensor values with same shape" do
    log =
      capture_log(fn ->
        for i <- 1..11 do
          t = Nx.tensor([i, i, i])
          fun = fn -> Nx.multiply(t, t) end
          Nx.Defn.jit_apply(fun, [], compiler: EXLA)
        end
      end)

    assert log =~ "EXLA has compiled"
    assert log =~ "same input shapes"
    assert log =~ "Pass changing tensors as explicit function arguments"
  end

  test "does not warn for stable functions" do
    log =
      capture_log(fn ->
        for _i <- 1..15 do
          t = Nx.tensor([1, 2, 3])
          Nx.Defn.jit_apply(&Nx.multiply/2, [t, t], compiler: EXLA)
        end
      end)

    refute log =~ "EXLA has compiled"
  end

  test "does not warn for different input shapes" do
    log =
      capture_log(fn ->
        for i <- 1..15 do
          t = Nx.iota({i})
          Nx.Defn.jit_apply(&Nx.multiply(&1, &1), [t], compiler: EXLA)
        end
      end)

    refute log =~ "EXLA has compiled"
  end
end
