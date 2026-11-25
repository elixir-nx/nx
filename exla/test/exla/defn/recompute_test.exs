defmodule EXLA.Defn.RecomputeTest do
  use EXLA.Case, async: true

  import Nx.Defn

  defn two_uses(x) do
    y = recompute(x, &Nx.exp/1)
    first = Nx.sum(y)
    Nx.sum(y * first)
  end

  defn two_uses_plain(x) do
    y = Nx.exp(x)
    first = Nx.sum(y)
    Nx.sum(y * first)
  end

  defn chain(x) do
    y = recompute(x, &Nx.exp/1)
    z = recompute(y, &Nx.tanh/1)
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
      recompute([row, column], fn row, column ->
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
    x = recompute([x, ws], fn x, ws -> pair(x, ws[0], ws[1]) end)
    x = recompute([x, ws], fn x, ws -> pair(x, ws[2], ws[3]) end)
    x = recompute([x, ws], fn x, ws -> pair(x, ws[4], ws[5]) end)
    x = recompute([x, ws], fn x, ws -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end

  defn tuple_two_uses(x) do
    {a, b} = recompute(x, fn x -> {Nx.exp(x), Nx.cos(x)} end)
    first = Nx.sum(a)
    Nx.sum(a * first) + Nx.sum(b)
  end

  defn tuple_two_uses_plain(x) do
    a = Nx.exp(x)
    first = Nx.sum(a)
    Nx.sum(a * first) + Nx.sum(Nx.cos(x))
  end

  defn used_in_cond(x) do
    y = recompute(x, &Nx.exp/1)
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
    y = recompute(x, &Nx.exp/1)
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
    y = recompute(x, &Nx.exp/1)
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

  describe "recompute/2" do
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

    test "computes inputs that are recompute blocks again" do
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
end
