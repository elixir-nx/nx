defmodule EXLA.Defn.RecomputeCheckpointTest do
  use EXLA.Case, async: true

  import Nx.Defn

  defn two_uses(x) do
    y = recompute_checkpoint(x, &Nx.exp/1)
    first = Nx.sum(y)
    Nx.sum(y * first)
  end

  defn two_uses_plain(x) do
    y = Nx.exp(x)
    first = Nx.sum(y)
    Nx.sum(y * first)
  end

  defn chain(x) do
    y = recompute_checkpoint(x, &Nx.exp/1)
    z = recompute_checkpoint(y, &Nx.tanh/1)
    first = Nx.sum(z)
    Nx.sum(z * first)
  end

  defn chain_plain(x) do
    z = x |> Nx.exp() |> Nx.tanh()
    first = Nx.sum(z)
    Nx.sum(z * first)
  end

  defn outer(row, column) do
    weights =
      recompute_checkpoint([row, column], fn row, column ->
        Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
      end)

    first = Nx.mean(weights, axes: [1])
    Nx.sum(weights * Nx.new_axis(Nx.sin(first), 1))
  end

  defn outer_plain(row, column) do
    weights = Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
    first = Nx.mean(weights, axes: [1])
    Nx.sum(weights * Nx.new_axis(Nx.sin(first), 1))
  end

  defn pair(x, w1, w2) do
    x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)
  end

  defn mlp_plain(ws, x) do
    x = pair(x, ws[0], ws[1])
    x = pair(x, ws[2], ws[3])
    x = pair(x, ws[4], ws[5])
    x = pair(x, ws[6], ws[7])
    Nx.sum(x)
  end

  defn mlp(ws, x) do
    x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[0], ws[1]) end)
    x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[2], ws[3]) end)
    x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[4], ws[5]) end)
    x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end

  defn tuple_two_uses(x) do
    {a, b} = recompute_checkpoint(x, fn x -> {Nx.exp(x), Nx.cos(x)} end)
    first = Nx.sum(a)
    Nx.sum(a * first) + Nx.sum(b)
  end

  defn tuple_two_uses_plain(x) do
    a = Nx.exp(x)
    first = Nx.sum(a)
    Nx.sum(a * first) + Nx.sum(Nx.cos(x))
  end

  defn used_in_cond(x) do
    y = recompute_checkpoint(x, &Nx.exp/1)
    first = Nx.sum(y)

    if first > 0 do
      Nx.sum(y * first)
    else
      first
    end
  end

  defn used_in_cond_plain(x) do
    y = Nx.exp(x)
    first = Nx.sum(y)

    if first > 0 do
      Nx.sum(y * first)
    else
      first
    end
  end

  defn while_initial(x) do
    y = recompute_checkpoint(x, &Nx.exp/1)
    first = Nx.sum(y)

    {_i, acc, _first} =
      while {i = 0, acc = y, first}, i < 3 do
        {i + 1, acc * first, first}
      end

    Nx.sum(acc)
  end

  defn while_initial_plain(x) do
    y = Nx.exp(x)
    first = Nx.sum(y)

    {_i, acc, _first} =
      while {i = 0, acc = y, first}, i < 3 do
        {i + 1, acc * first, first}
      end

    Nx.sum(acc)
  end

  defn block_consumer(x) do
    y = recompute_checkpoint(x, &Nx.exp/1)
    first = Nx.sum(y)
    Nx.sum(Nx.take(y, Nx.as_type(Nx.remainder(first, 4), :s32)) * first)
  end

  defn block_consumer_plain(x) do
    y = Nx.exp(x)
    first = Nx.sum(y)
    Nx.sum(Nx.take(y, Nx.as_type(Nx.remainder(first, 4), :s32)) * first)
  end

  defp count(hlo, needle), do: hlo |> String.split(needle) |> length() |> Kernel.-(1)

  defp optimized(fun, args),
    do: fun |> EXLA.to_executable(args) |> EXLA.Executable.optimized_hlo()

  defp scratch_bytes(fun, args) do
    %{temp_size_in_bytes: bytes} =
      fun |> EXLA.to_executable(args) |> EXLA.Executable.memory_stats()

    bytes
  end

  describe "recompute_checkpoint/2" do
    test "returns the value of the function" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)
      assert_all_close(two_uses(x), two_uses_plain(x))
      assert_all_close(chain(x), chain_plain(x))
    end

    test "computes the result again for a later use" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

      assert count(optimized(&two_uses_plain/1, [x]), "exponential(") == 1
      assert count(optimized(&two_uses/1, [x]), "exponential(") == 2
    end

    test "computes inputs that are recompute_checkpoint blocks again" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)
      hlo = optimized(&chain/1, [x])

      assert count(hlo, "tanh(") == 2
      assert count(hlo, "exponential(") == 2
    end

    test "computes a tuple result again for a later use of an element" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

      assert_all_close(tuple_two_uses(x), tuple_two_uses_plain(x))
      assert count(optimized(&tuple_two_uses_plain/1, [x]), "exponential(") == 1
      assert count(optimized(&tuple_two_uses/1, [x]), "exponential(") == 2
    end

    test "computes the result again for a use inside cond" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

      assert_all_close(used_in_cond(x), used_in_cond_plain(x))
      assert count(optimized(&used_in_cond_plain/1, [x]), "exponential(") == 1
      assert count(optimized(&used_in_cond/1, [x]), "exponential(") == 2
    end

    test "computes the result again for a while that starts from it" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

      assert_all_close(while_initial(x), while_initial_plain(x))
      assert count(optimized(&while_initial_plain/1, [x]), "exponential(") == 1
      assert count(optimized(&while_initial/1, [x]), "exponential(") == 2
    end

    test "computes the result again for a block that reads it" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

      assert_all_close(block_consumer(x), block_consumer_plain(x))
      assert count(optimized(&block_consumer_plain/1, [x]), "exponential(") == 1
      assert count(optimized(&block_consumer/1, [x]), "exponential(") == 2
    end

    # On CUDA the plain function already fuses the matrix into both uses and
    # never holds it, so there is nothing to save.
    @tag :skip_cuda
    test "does not keep a large result between two uses" do
      row = Nx.iota({512}, type: :f32) |> Nx.divide(512)
      matrix_bytes = 512 * 512 * 4

      assert_all_close(outer(row, row), outer_plain(row, row))
      assert scratch_bytes(&outer_plain/2, [row, row]) >= matrix_bytes
      assert scratch_bytes(&outer/2, [row, row]) < matrix_bytes
    end

    test "the gradient of a chain matches the plain function" do
      {ws, _} = Nx.Random.normal(Nx.Random.key(0), shape: {8, 16, 16}, type: :f32)
      x = Nx.iota({8, 16}, type: :f32) |> Nx.divide(128)

      assert_all_close(
        Nx.Defn.grad(ws, &mlp(&1, x)),
        Nx.Defn.grad(ws, &mlp_plain(&1, x))
      )
    end

    test "lowers the peak scratch memory of a gradient and runs more matrix products" do
      n = 256
      batch = 65536
      ws = Nx.broadcast(Nx.tensor(0.01, type: :f32), {8, n, n})
      x = Nx.broadcast(Nx.tensor(0.5, type: :f32), {batch, n})

      with_recompute =
        EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp(&1, x)) end, [ws, x])

      plain = EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp_plain(&1, x)) end, [ws, x])

      %{temp_size_in_bytes: with_temp} = EXLA.Executable.memory_stats(with_recompute)
      %{temp_size_in_bytes: plain_temp} = EXLA.Executable.memory_stats(plain)

      activation_bytes = batch * n * 4
      assert plain_temp - with_temp >= activation_bytes

      # Measured on the host client: 1009 MB and 23 products plain, 807 MB and
      # 47 products with recompute. On CUDA: 604 MB plain, 269 MB with.
      products = &count(EXLA.Executable.optimized_hlo(&1), " dot(")
      assert products.(with_recompute) > products.(plain)
    end
  end

  defn nested(x) do
    out =
      recompute_checkpoint(x, fn x ->
        y = x * x
        z = recompute_checkpoint(y, fn y -> Nx.exp(y) end)
        Nx.sin(z)
      end)

    Nx.sum(out * out)
  end

  defn nested_plain(x) do
    out = Nx.sin(Nx.exp(x * x))
    Nx.sum(out * out)
  end

  defn vectorized_loss(x, w) do
    Nx.sum(recompute_checkpoint([x, w], fn x, w -> Nx.sin(Nx.dot(x, w)) * x end))
  end

  defn vectorized_loss_plain(x, w) do
    Nx.sum(Nx.sin(Nx.dot(x, w)) * x)
  end

  test "nested recompute_checkpoint blocks compute the same gradient as the plain function" do
    x = Nx.tensor([0.5, 1.0, 1.5])
    assert_all_close(Nx.Defn.grad(x, &nested/1), Nx.Defn.grad(x, &nested_plain/1))
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

  defn outer_in_cond(row, column) do
    weights =
      recompute_checkpoint([row, column], fn row, column ->
        Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
      end)

    first = Nx.mean(weights, axes: [1])

    if Nx.sum(first) > 0 do
      Nx.sum(weights * Nx.new_axis(Nx.sin(first), 1))
    else
      Nx.sum(first)
    end
  end

  defn outer_in_cond_plain(row, column) do
    weights = Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
    first = Nx.mean(weights, axes: [1])

    if Nx.sum(first) > 0 do
      Nx.sum(weights * Nx.new_axis(Nx.sin(first), 1))
    else
      Nx.sum(first)
    end
  end

  defn outer_in_while(row, column) do
    weights =
      recompute_checkpoint([row, column], fn row, column ->
        Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
      end)

    first = Nx.mean(weights, axes: [1])

    {_i, total, _row, _column} =
      while {i = 0, total = Nx.sum(first), row, column}, i < 3 do
        weights =
          recompute_checkpoint([row, column], fn row, column ->
            Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
          end)

        scale = Nx.mean(weights, axes: [1])
        {i + 1, total + Nx.sum(weights * Nx.new_axis(Nx.sin(scale), 1)), row, column}
      end

    total
  end

  defn outer_in_while_plain(row, column) do
    weights = Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
    first = Nx.mean(weights, axes: [1])

    {_i, total, _row, _column} =
      while {i = 0, total = Nx.sum(first), row, column}, i < 3 do
        weights = Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
        scale = Nx.mean(weights, axes: [1])
        {i + 1, total + Nx.sum(weights * Nx.new_axis(Nx.sin(scale), 1)), row, column}
      end

    total
  end

  # On CUDA the plain function already fuses the matrix into both uses and
  # never holds it, so these two are checked on the host client only.
  @tag :skip_cuda
  test "does not keep a large result for a later use inside cond" do
    row = Nx.iota({512}, type: :f32) |> Nx.divide(512)
    matrix_bytes = 512 * 512 * 4

    assert_all_close(outer_in_cond(row, row), outer_in_cond_plain(row, row))
    assert scratch_bytes(&outer_in_cond_plain/2, [row, row]) >= matrix_bytes
    assert scratch_bytes(&outer_in_cond/2, [row, row]) < matrix_bytes
  end

  @tag :skip_cuda
  test "does not keep a large result between two uses inside a while body" do
    row = Nx.iota({512}, type: :f32) |> Nx.divide(512)
    matrix_bytes = 512 * 512 * 4

    assert_all_close(outer_in_while(row, row), outer_in_while_plain(row, row))
    assert scratch_bytes(&outer_in_while_plain/2, [row, row]) >= matrix_bytes
    assert scratch_bytes(&outer_in_while/2, [row, row]) < matrix_bytes
  end

  defn mlp_in_while_plain(ws, x, iterations) do
    {_i, x, _ws, _iterations} =
      while {i = 0, x, ws, iterations}, i < iterations do
        x = pair(x, ws[0], ws[1])
        x = pair(x, ws[2], ws[3])
        x = pair(x, ws[4], ws[5])
        x = pair(x, ws[6], ws[7])
        {i + 1, x, ws, iterations}
      end

    Nx.sum(x)
  end

  defn mlp_in_while(ws, x, iterations) do
    {_i, x, _ws, _iterations} =
      while {i = 0, x, ws, iterations}, i < iterations do
        x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[0], ws[1]) end)
        x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[2], ws[3]) end)
        x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[4], ws[5]) end)
        x = recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[6], ws[7]) end)
        {i + 1, x, ws, iterations}
      end

    Nx.sum(x)
  end

  defn one_pair_in_while_plain(ws, x, iterations) do
    {_i, x, _ws, _iterations} =
      while {i = 0, x, ws, iterations}, i < iterations do
        {i + 1, pair(x, ws[0], ws[1]), ws, iterations}
      end

    Nx.sum(x)
  end

  defn one_pair_in_while(ws, x, iterations) do
    {_i, x, _ws, _iterations} =
      while {i = 0, x, ws, iterations}, i < iterations do
        {i + 1, recompute_checkpoint([x, ws], fn x, ws -> pair(x, ws[0], ws[1]) end), ws,
         iterations}
      end

    Nx.sum(x)
  end

  # The gradient of a while keeps no state per iteration. It rebuilds each
  # state from the initial one and holds the intermediates of one iteration
  # at a time, so the only thing left to save is what one iteration keeps.
  #
  # Measured on the host client, in MB of scratch: 1279 plain and 1080 with
  # recompute for four blocks per iteration, 472 for both with one block.
  describe "the gradient of a while" do
    setup do
      n = 256
      batch = 65536

      %{
        ws: Nx.broadcast(Nx.tensor(0.01, type: :f32), {8, n, n}),
        x: Nx.broadcast(Nx.tensor(0.5, type: :f32), {batch, n}),
        activation_bytes: batch * n * 4
      }
    end

    test "needs less scratch memory when one iteration is a chain of blocks", context do
      %{ws: ws, x: x, activation_bytes: activation_bytes} = context

      scratch = fn fun ->
        scratch_bytes(fn ws, x, n -> Nx.Defn.grad(ws, &fun.(&1, x, n)) end, [ws, x, 4])
      end

      assert scratch.(&mlp_in_while_plain/3) - scratch.(&mlp_in_while/3) >= activation_bytes
    end

    test "needs the same scratch memory when one iteration is a single block", context do
      %{ws: ws, x: x, activation_bytes: activation_bytes} = context

      scratch = fn fun ->
        scratch_bytes(fn ws, x, n -> Nx.Defn.grad(ws, &fun.(&1, x, n)) end, [ws, x, 4])
      end

      difference = scratch.(&one_pair_in_while_plain/3) - scratch.(&one_pair_in_while/3)
      assert abs(difference) < activation_bytes
    end

    test "matches the plain function" do
      {ws, _} = Nx.Random.normal(Nx.Random.key(0), shape: {8, 16, 16}, type: :f32)
      ws = Nx.multiply(ws, 0.2)
      x = Nx.iota({8, 16}, type: :f32) |> Nx.divide(128)

      assert_all_close(
        Nx.Defn.grad(ws, &mlp_in_while(&1, x, 3)),
        Nx.Defn.grad(ws, &mlp_in_while_plain(&1, x, 3))
      )
    end
  end
end
