defmodule EXLA.Defn.CheckpointTest do
  use EXLA.Case, async: true

  import Nx.Defn

  defn body(x), do: x |> Nx.exp() |> Nx.sin()

  defn loss_with_checkpoint(x) do
    y = checkpoint(x, &body/1)
    Nx.sum(y * y)
  end

  defn loss_without_checkpoint(x) do
    y = body(x)
    Nx.sum(y * y)
  end

  defn pair(x, w1, w2) do
    x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)
  end

  defn mlp_without_checkpoint(ws, x) do
    x = pair(x, ws[0], ws[1])
    x = pair(x, ws[2], ws[3])
    x = pair(x, ws[4], ws[5])
    x = pair(x, ws[6], ws[7])
    Nx.sum(x)
  end

  defn mlp_with_checkpoints(ws, x) do
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[0], ws[1]) end)
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[2], ws[3]) end)
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[4], ws[5]) end)
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end

  defp count(hlo, needle), do: hlo |> String.split(needle) |> length() |> Kernel.-(1)

  defn nested(x) do
    out =
      checkpoint(x, fn x ->
        y = x * x
        z = checkpoint(y, fn y -> Nx.exp(y) end)
        Nx.sin(z)
      end)

    Nx.sum(out * out)
  end

  defn nested_plain(x) do
    out = Nx.sin(Nx.exp(x * x))
    Nx.sum(out * out)
  end

  defn cond_inside(x) do
    checkpoint(x, fn x ->
      if Nx.sum(x) > 0, do: Nx.sum(Nx.sin(x)), else: Nx.sum(Nx.cos(x))
    end)
  end

  defn cond_outside(x) do
    if Nx.sum(x) > 0 do
      checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end)
    else
      Nx.sum(Nx.cos(x))
    end
  end

  defn cond_plain(x) do
    if Nx.sum(x) > 0, do: Nx.sum(Nx.sin(x)), else: Nx.sum(Nx.cos(x))
  end

  defn while_inside(x) do
    {_, acc} =
      while {i = 0, acc = x}, i < 3 do
        {i + 1, checkpoint(acc, fn acc -> Nx.sin(acc) * acc end)}
      end

    Nx.sum(acc)
  end

  defn while_plain(x) do
    {_, acc} =
      while {i = 0, acc = x}, i < 3 do
        {i + 1, Nx.sin(acc) * acc}
      end

    Nx.sum(acc)
  end

  defn vectorized_loss(x, w) do
    Nx.sum(checkpoint([x, w], fn x, w -> Nx.sin(Nx.dot(x, w)) * x end))
  end

  defn vectorized_loss_plain(x, w) do
    Nx.sum(Nx.sin(Nx.dot(x, w)) * x)
  end

  defn pair_loss(pair) do
    {a, b} = checkpoint(pair, fn {a, b} -> {Nx.sin(a), Nx.cos(b)} end)
    Nx.sum(a * b)
  end

  defn pair_loss_plain({a, b}) do
    Nx.sum(Nx.sin(a) * Nx.cos(b))
  end

  defn captured_loss(x, w) do
    Nx.sum(checkpoint(x, fn x -> Nx.sin(Nx.dot(x, w)) * x end))
  end

  defn captured_vectorized_loss(x, scale) do
    Nx.sum(checkpoint(x, fn x -> Nx.multiply(Nx.sin(x), scale) end))
  end

  defn captured_vectorized_loss_plain(x, scale) do
    Nx.sum(Nx.multiply(Nx.sin(x), scale))
  end

  test "computes the same gradient as the plain function" do
    x = Nx.iota({16}, type: :f32) |> Nx.divide(16)

    assert_all_close(
      Nx.Defn.grad(x, &loss_with_checkpoint/1),
      Nx.Defn.grad(x, &loss_without_checkpoint/1)
    )
  end

  test "checkpoints with equal shapes and different bodies keep their own body" do
    {ws, _} = Nx.Random.normal(Nx.Random.key(0), shape: {8, 16, 16}, type: :f32)
    x = Nx.iota({8, 16}, type: :f32) |> Nx.divide(128)

    assert_equal(
      Nx.Defn.grad(ws, &mlp_with_checkpoints(&1, x)),
      Nx.Defn.grad(ws, &mlp_without_checkpoint(&1, x))
    )
  end

  @tag :rematerialization
  test "keeps the recomputed body in the compiled gradient" do
    x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

    with_checkpoint =
      EXLA.to_executable(fn x -> Nx.Defn.grad(x, &loss_with_checkpoint/1) end, [x])

    without_checkpoint =
      EXLA.to_executable(fn x -> Nx.Defn.grad(x, &loss_without_checkpoint/1) end, [x])

    assert count(EXLA.Executable.optimized_hlo(without_checkpoint), "exponential(") == 1
    assert count(EXLA.Executable.optimized_hlo(with_checkpoint), "exponential(") == 2
  end

  @tag :rematerialization
  test "lowers the peak scratch memory of the gradient by at least one activation" do
    # The activation has to be much larger than the workspaces the autotuner
    # may pick for the matrix multiplications, which vary from run to run.
    n = 256
    batch = 65536
    ws = Nx.broadcast(Nx.tensor(0.01, type: :f32), {8, n, n})
    x = Nx.broadcast(Nx.tensor(0.5, type: :f32), {batch, n})

    with_checkpoint =
      EXLA.to_executable(
        fn ws, x -> Nx.Defn.grad(ws, &mlp_with_checkpoints(&1, x)) end,
        [ws, x]
      )

    without_checkpoint =
      EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp_without_checkpoint(&1, x)) end, [
        ws,
        x
      ])

    %{temp_size_in_bytes: with_temp} = EXLA.Executable.memory_stats(with_checkpoint)
    %{temp_size_in_bytes: without_temp} = EXLA.Executable.memory_stats(without_checkpoint)

    # Each checkpointed pair of layers drops one f32 activation of {batch, n}
    # from the saved set. The measured saving is about two of them.
    activation_bytes = batch * n * 4
    assert without_temp - with_temp >= activation_bytes
  end

  test "nested checkpoints compute the same gradient as the plain function" do
    x = Nx.tensor([0.5, 1.0, 1.5])
    assert_all_close(Nx.Defn.grad(x, &nested/1), Nx.Defn.grad(x, &nested_plain/1))
  end

  test "a cond inside or around a checkpoint computes the same gradient" do
    for x <- [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([-1.0, -2.0, -3.0])] do
      plain = Nx.Defn.grad(x, &cond_plain/1)
      assert_all_close(Nx.Defn.grad(x, &cond_inside/1), plain)
      assert_all_close(Nx.Defn.grad(x, &cond_outside/1), plain)
    end
  end

  test "a checkpoint inside a while body computes the same gradient" do
    x = Nx.tensor([0.5, 1.0])
    assert_all_close(Nx.Defn.grad(x, &while_inside/1), Nx.Defn.grad(x, &while_plain/1))
  end

  test "a vectorized activation keeps its axes in the value and the gradient" do
    x = Nx.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]) |> Nx.vectorize(:batch)
    w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])

    value = vectorized_loss(x, w)
    assert value.vectorized_axes == [batch: 3]
    assert_all_close(value, vectorized_loss_plain(x, w))

    gradient = Nx.Defn.grad(x, &vectorized_loss(&1, w))
    assert gradient.vectorized_axes == [batch: 3]
    assert_all_close(gradient, Nx.Defn.grad(x, &vectorized_loss_plain(&1, w)))
  end

  test "a tuple input is flattened into the block and rebuilt for the body" do
    pair = {Nx.tensor([0.5, 1.0, 1.5]), Nx.tensor([2.0, 2.5, 3.0])}
    assert_all_close(pair_loss(pair), pair_loss_plain(pair))
    {grad_a, grad_b} = Nx.Defn.grad(pair, &pair_loss/1)
    {plain_a, plain_b} = Nx.Defn.grad(pair, &pair_loss_plain/1)
    assert_all_close(grad_a, plain_a)
    assert_all_close(grad_b, plain_b)
  end

  test "a weight captured by the body gets the same gradient as an explicit input" do
    x = Nx.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
    w = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])

    assert_all_close(captured_loss(x, w), vectorized_loss(x, w))

    assert_all_close(
      Nx.Defn.grad(w, &captured_loss(x, &1)),
      Nx.Defn.grad(w, &vectorized_loss(x, &1))
    )
  end

  test "a captured vectorized tensor keeps its axes in the value and the gradient" do
    x = Nx.tensor([0.5, 1.0])
    scale = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) |> Nx.vectorize(:batch)

    value = captured_vectorized_loss(x, scale)
    assert value.vectorized_axes == [batch: 3]
    assert_all_close(value, captured_vectorized_loss_plain(x, scale))

    gradient = Nx.Defn.grad(scale, &captured_vectorized_loss(x, &1))
    assert gradient.vectorized_axes == [batch: 3]
    assert_all_close(gradient, Nx.Defn.grad(scale, &captured_vectorized_loss_plain(x, &1)))
  end

  @tag :rematerialization
  test "the gradient recomputes the exponential it needs and nothing else" do
    x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

    with_checkpoint =
      EXLA.to_executable(fn x -> Nx.Defn.grad(x, &loss_with_checkpoint/1) end, [x])

    without_checkpoint =
      EXLA.to_executable(fn x -> Nx.Defn.grad(x, &loss_without_checkpoint/1) end, [x])

    %{"transcendentals" => with_count} = EXLA.Executable.cost_analysis(with_checkpoint)
    %{"transcendentals" => without_count} = EXLA.Executable.cost_analysis(without_checkpoint)

    # The gradient of sin needs exp(x) again, so the recomputation adds one
    # exponential per element. The recomputed sin feeds nothing and the
    # compiler removes it.
    assert with_count - without_count == 1024
  end

  @tag :rematerialization
  test "a nested checkpoint recomputes its body once per enclosing recomputation" do
    x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

    nested = EXLA.to_executable(fn x -> Nx.Defn.grad(x, &nested/1) end, [x])
    plain = EXLA.to_executable(fn x -> Nx.Defn.grad(x, &nested_plain/1) end, [x])

    assert count(EXLA.Executable.optimized_hlo(plain), "exponential(") == 1
    assert count(EXLA.Executable.optimized_hlo(nested), "exponential(") == 3
  end
end
