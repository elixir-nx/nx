defmodule Nx.Defn.CheckpointTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Testing

  # The evaluator calls these with tensors, tracing calls them with
  # expressions. Only the evaluator calls are counted.
  def counted_exp(x), do: counted(:exp, x, &Nx.exp/1)
  def counted_sin(y), do: counted(:sin, y, &Nx.sin/1)

  defp counted(name, %Nx.Tensor{data: %Nx.Defn.Expr{}} = x, fun) when is_atom(name), do: fun.(x)

  defp counted(name, x, fun) do
    send(self(), {:ran, name})
    fun.(x)
  end

  defp runs(name) do
    receive do
      {:ran, ^name} -> 1 + runs(name)
    after
      0 -> 0
    end
  end

  defn two_uses(x) do
    y = checkpoint(x, &counted_exp/1)
    Nx.sum(y) + Nx.reduce_max(y)
  end

  defn two_uses_plain(x) do
    y = Nx.exp(x)
    Nx.sum(y) + Nx.reduce_max(y)
  end

  defn chain(x) do
    y = checkpoint(x, &counted_exp/1)
    z = checkpoint(y, &counted_sin/1)
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn chain_plain(x) do
    z = x |> Nx.exp() |> Nx.sin()
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn dense(x, w1, w2), do: x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.tanh()

  defn mlp(weights, x) do
    y = checkpoint([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
    z = checkpoint([y, weights], fn y, weights -> dense(y, weights.c, weights.d) end)
    Nx.sum(z * z)
  end

  defn mlp_plain(weights, x) do
    z = x |> dense(weights.a, weights.b) |> dense(weights.c, weights.d)
    Nx.sum(z * z)
  end

  defn tuple_output(x) do
    {a, b} = checkpoint(x, fn x -> {Nx.sin(x), Nx.cos(x)} end)
    Nx.sum(a * b)
  end

  defn tuple_output_plain(x), do: Nx.sum(Nx.sin(x) * Nx.cos(x))

  defp weights do
    key = Nx.Random.key(7)
    {a, key} = Nx.Random.normal(key, shape: {3, 4})
    {b, key} = Nx.Random.normal(key, shape: {4, 3})
    {c, key} = Nx.Random.normal(key, shape: {3, 4})
    {d, _key} = Nx.Random.normal(key, shape: {4, 3})
    %{a: a, b: b, c: c, d: d}
  end

  describe "checkpoint/2" do
    test "returns the value of the function" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(two_uses(x), two_uses_plain(x))
      assert_all_close(chain(x), chain_plain(x))
    end

    test "runs the function once for every use of the result" do
      two_uses(Nx.tensor([0.1, 0.2, 0.3]))
      assert runs(:exp) == 2
    end

    # The evaluator keeps the evaluated inputs of a checkpoint, so the first
    # checkpoint of a chain runs once. EXLA lowers it again for every use.
    test "the evaluator keeps the inputs of a checkpoint" do
      chain(Nx.tensor([0.1, 0.2, 0.3]))
      assert runs(:sin) == 2
      assert runs(:exp) == 1
    end

    test "the gradient of a chain matches the plain function" do
      x = Nx.iota({2, 3}, type: :f32) |> Nx.divide(6)
      weights = weights()

      grads = Nx.Defn.grad(weights, &mlp(&1, x))
      plain_grads = Nx.Defn.grad(weights, &mlp_plain(&1, x))

      for name <- [:a, :b, :c, :d] do
        assert_all_close(grads[name], plain_grads[name])
      end

      assert_all_close(
        Nx.Defn.grad(x, &mlp(weights, &1)),
        Nx.Defn.grad(x, &mlp_plain(weights, &1))
      )
    end

    test "returns a tuple of tensors" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(tuple_output(x), tuple_output_plain(x))

      assert_all_close(
        Nx.Defn.grad(x, &tuple_output/1),
        Nx.Defn.grad(x, &tuple_output_plain/1)
      )
    end

    test "is the function itself outside defn" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(Nx.Defn.Kernel.checkpoint(x, &Nx.exp/1), Nx.exp(x))
    end

    test "raises when the function arity does not match the inputs" do
      assert_raise ArgumentError, ~r/expected a function of arity 2/, fn ->
        Nx.Defn.Kernel.checkpoint([Nx.tensor(1), Nx.tensor(2)], &Nx.exp/1)
      end
    end
  end
end
