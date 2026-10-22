defmodule EXLA.ZeroSizeTest do
  use EXLA.Case, async: true

  # Each case runs on Nx.BinaryBackend for the reference result, then on EXLA.Backend and
  # through the EXLA compiler.
  test "zero-size cases on the backend and through the compiler" do
    cases = [
      {"add", &Nx.add(&1, 1), [{2, 0}]},
      {"broadcast {1} to {0}", &Nx.add(&1, Nx.iota({2, 1})), [{2, 0}]},
      {"exp", &Nx.exp/1, [{2, 0}]},
      {"reshape", &Nx.reshape(&1, {0, 5}), [{0}]},
      {"transpose", &Nx.transpose(&1, axes: [2, 0, 1]), [{2, 0, 3}]},
      {"concatenate", &Nx.concatenate([&1, Nx.iota({2, 3}, type: :f32)], axis: 1), [{2, 0}]},
      {"slice", &Nx.slice(&1, [0, 0], [2, 0]), [{2, 3}]},
      {"pad", &Nx.pad(&1, 1.0, [{0, 0, 0}, {1, 1, 0}]), [{2, 0}]},
      {"sum", &Nx.sum(&1, axes: [1]), [{2, 0}]},
      {"product", &Nx.product(&1, axes: [1]), [{2, 0}]},
      {"mean", &Nx.mean(&1, axes: [1]), [{2, 0}]},
      {"reduce_max", &Nx.reduce_max(&1, axes: [1]), [{2, 0}]},
      {"reduce_min", &Nx.reduce_min(&1, axes: [1]), [{2, 0}]},
      {"argmax", &Nx.argmax(&1, axis: 1), [{2, 0}]},
      {"argmin", &Nx.argmin(&1, axis: 1), [{2, 0}]},
      {"logsumexp", &Nx.logsumexp(&1, axes: [1]), [{2, 0}]},
      {"cumulative_sum", &Nx.cumulative_sum(&1, axis: 1), [{2, 0}]},
      {"dot over an empty axis", &Nx.dot/2, [{2, 0}, {0, 3}]},
      {"dot with an empty output", &Nx.dot/2, [{0, 3}, {3, 2}]},
      {"conv with empty channels", &Nx.conv/2, [{1, 0, 4}, {2, 0, 2}]},
      {"conv with an empty batch", &Nx.conv/2, [{0, 1, 4}, {1, 1, 2}]},
      {"window_sum", &Nx.window_sum(&1, {1, 2}), [{0, 4}]},
      {"sort", &Nx.sort(&1, axis: 1), [{2, 0}]},
      {"take", &Nx.take(&1, Nx.tensor([], type: :s64), axis: 1), [{2, 3}]},
      {"gather", &Nx.gather(&1, empty({0, 2}, :s64)), [{2, 3}]},
      {"indexed_add", &Nx.indexed_add(&1, empty({0, 2}, :s64), empty({0}, :f32)), [{2, 3}]},
      {"indexed_put", &Nx.indexed_put(&1, empty({0, 2}, :s64), empty({0}, :f32)), [{2, 3}]}
    ]

    for {name, fun, shapes} <- cases do
      expected = apply(fun, inputs(shapes, Nx.BinaryBackend))
      eager = apply(fun, inputs(shapes, EXLA.Backend))
      compiled = apply(Nx.Defn.jit(fun, compiler: EXLA), inputs(shapes, EXLA.Backend))

      assert_all_close(eager, expected, equal_nan: true, input: "#{name}, eager")
      assert_all_close(compiled, expected, equal_nan: true, input: "#{name}, compiled")
    end
  end

  test "grad through a broadcast to a zero-size shape" do
    fun = Nx.Defn.grad(&Nx.sum(Nx.multiply(Nx.broadcast(&1, {0}), 3)))
    assert_equal(Nx.Defn.jit(fun, compiler: EXLA).(Nx.tensor(2.0)), Nx.tensor(0.0))
  end

  defp inputs(shapes, backend) do
    for shape <- shapes do
      shape |> Nx.iota(type: :f32, backend: backend) |> Nx.divide(10)
    end
  end

  defp empty(shape, type), do: Nx.broadcast(Nx.tensor(0, type: type), shape)
end
