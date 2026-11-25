defmodule Nx.Defn.RecomputeTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Testing

  # The evaluator calls these with tensors, tracing calls them with
  # expressions. Only the evaluator calls are counted.
  def counted_exp(x), do: counted(:exp, x, &Nx.exp/1)
  def counted_sin(y), do: counted(:sin, y, &Nx.sin/1)
  def counted_exp_cos(x), do: {counted_exp(x), Nx.cos(x)}

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
    y = recompute(x, &counted_exp/1)
    Nx.sum(y) + Nx.reduce_max(y)
  end

  defn two_uses_plain(x) do
    y = Nx.exp(x)
    Nx.sum(y) + Nx.reduce_max(y)
  end

  defn chain(x) do
    y = recompute(x, &counted_exp/1)
    z = recompute(y, &counted_sin/1)
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn chain_plain(x) do
    z = x |> Nx.exp() |> Nx.sin()
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn dense(x, w1, w2), do: x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.tanh()

  defn mlp(weights, x) do
    y = recompute([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
    z = recompute([y, weights], fn y, weights -> dense(y, weights.c, weights.d) end)
    Nx.sum(z * z)
  end

  defn mlp_plain(weights, x) do
    z = x |> dense(weights.a, weights.b) |> dense(weights.c, weights.d)
    Nx.sum(z * z)
  end

  defn tuple_output(x) do
    {a, b} = recompute(x, fn x -> {Nx.sin(x), Nx.cos(x)} end)
    Nx.sum(a * b)
  end

  defn tuple_output_plain(x), do: Nx.sum(Nx.sin(x) * Nx.cos(x))

  defn tuple_two_uses(x) do
    {a, b} = recompute(x, &counted_exp_cos/1)
    Nx.sum(a) + Nx.reduce_max(a) + Nx.sum(b)
  end

  defn tuple_two_uses_plain(x) do
    a = Nx.exp(x)
    Nx.sum(a) + Nx.reduce_max(a) + Nx.sum(Nx.cos(x))
  end

  defn used_in_cond(x) do
    y = recompute(x, &counted_exp/1)

    if Nx.sum(y) > 0 do
      Nx.reduce_max(y) * 2
    else
      Nx.sum(x)
    end
  end

  defn used_in_cond_plain(x) do
    y = Nx.exp(x)

    if Nx.sum(y) > 0 do
      Nx.reduce_max(y) * 2
    else
      Nx.sum(x)
    end
  end

  defn used_in_while(x) do
    y = recompute(x, &counted_exp/1)

    {_i, total, _y} =
      while {i = 0, total = Nx.sum(y), y}, i < 3 do
        {i + 1, total + Nx.sum(y * total), y}
      end

    total
  end

  defn used_in_while_plain(x) do
    y = Nx.exp(x)

    {_i, total, _y} =
      while {i = 0, total = Nx.sum(y), y}, i < 3 do
        {i + 1, total + Nx.sum(y * total), y}
      end

    total
  end

  defn while_initial(x) do
    y = recompute(x, &counted_exp/1)

    {_i, acc} =
      while {i = 0, acc = y}, i < 3 do
        {i + 1, acc * 0.5}
      end

    Nx.sum(acc) + Nx.sum(y)
  end

  defn while_initial_plain(x) do
    y = Nx.exp(x)

    {_i, acc} =
      while {i = 0, acc = y}, i < 3 do
        {i + 1, acc * 0.5}
      end

    Nx.sum(acc) + Nx.sum(y)
  end

  defp weights do
    key = Nx.Random.key(7)
    {a, key} = Nx.Random.normal(key, shape: {3, 4})
    {b, key} = Nx.Random.normal(key, shape: {4, 3})
    {c, key} = Nx.Random.normal(key, shape: {3, 4})
    {d, _key} = Nx.Random.normal(key, shape: {4, 3})
    %{a: a, b: b, c: c, d: d}
  end

  describe "recompute/2" do
    test "returns the value of the function" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(two_uses(x), two_uses_plain(x))
      assert_all_close(chain(x), chain_plain(x))
    end

    test "runs the function once for every use of the result" do
      two_uses(Nx.tensor([0.1, 0.2, 0.3]))
      assert runs(:exp) == 2
    end

    test "computes inputs that are recompute blocks again" do
      chain(Nx.tensor([0.1, 0.2, 0.3]))
      assert runs(:sin) == 2
      assert runs(:exp) == 2
    end

    test "runs a function that returns a tuple once for every use of an element" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(tuple_two_uses(x), tuple_two_uses_plain(x))
      assert runs(:exp) == 3
    end

    test "runs the function again for a use inside cond" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(used_in_cond(x), used_in_cond_plain(x))
      assert runs(:exp) == 2

      assert_all_close(Nx.Defn.grad(x, &used_in_cond/1), Nx.Defn.grad(x, &used_in_cond_plain/1))
    end

    # The loop state is a value, so the body of the loop reads the copy made
    # for the while and does not run the function again.
    test "a while carries one copy through its iterations" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(used_in_while(x), used_in_while_plain(x))
      assert runs(:exp) == 2

      assert_all_close(
        Nx.Defn.grad(x, &used_in_while/1),
        Nx.Defn.grad(x, &used_in_while_plain/1)
      )
    end

    test "is the initial value of a while" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(while_initial(x), while_initial_plain(x))
      assert runs(:exp) == 2
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
      assert_all_close(Nx.Defn.Kernel.recompute(x, &Nx.exp/1), Nx.exp(x))
    end

    test "raises when the function returns a container other than a tuple" do
      x = Nx.tensor([1.0, 2.0])

      assert_raise ArgumentError, ~r/must return a tensor or a tuple of tensors/, fn ->
        Nx.Defn.jit(fn x -> Nx.Defn.Kernel.recompute(x, fn x -> %{value: x} end) end).(x)
      end
    end

    test "raises when the function arity does not match the inputs" do
      assert_raise ArgumentError, ~r/expected a function of arity 2/, fn ->
        Nx.Defn.Kernel.recompute([Nx.tensor(1), Nx.tensor(2)], &Nx.exp/1)
      end
    end
  end
end
