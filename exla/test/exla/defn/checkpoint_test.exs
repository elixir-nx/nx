defmodule EXLA.Defn.CheckpointTest do
  use EXLA.Case, async: true

  import Nx.Defn

  defn two_uses(x) do
    y = checkpoint(x, &Nx.exp/1)
    first = Nx.sum(y)
    Nx.sum(y * first)
  end

  defn two_uses_plain(x) do
    y = Nx.exp(x)
    first = Nx.sum(y)
    Nx.sum(y * first)
  end

  defn chain(x) do
    y = checkpoint(x, &Nx.exp/1)
    z = checkpoint(y, &Nx.tanh/1)
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
      checkpoint([row, column], fn row, column ->
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
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[0], ws[1]) end)
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[2], ws[3]) end)
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[4], ws[5]) end)
    x = checkpoint([x, ws], fn x, ws -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end

  defp count(hlo, needle), do: hlo |> String.split(needle) |> length() |> Kernel.-(1)

  defp optimized(fun, args),
    do: fun |> EXLA.to_executable(args) |> EXLA.Executable.optimized_hlo()

  defp scratch_bytes(fun, args) do
    %{temp_size_in_bytes: bytes} =
      fun |> EXLA.to_executable(args) |> EXLA.Executable.memory_stats()

    bytes
  end

  describe "checkpoint/2" do
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

    test "computes the checkpoint inputs of a checkpoint again" do
      x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)
      hlo = optimized(&chain/1, [x])

      assert count(hlo, "tanh(") == 2
      assert count(hlo, "exponential(") == 2
    end

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

      with_checkpoint =
        EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp(&1, x)) end, [ws, x])

      plain = EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp_plain(&1, x)) end, [ws, x])

      %{temp_size_in_bytes: with_temp} = EXLA.Executable.memory_stats(with_checkpoint)
      %{temp_size_in_bytes: plain_temp} = EXLA.Executable.memory_stats(plain)

      activation_bytes = batch * n * 4
      assert plain_temp - with_temp >= activation_bytes

      # Measured on the host client: 23 products plain, 47 with checkpoints.
      products = &count(EXLA.Executable.optimized_hlo(&1), " dot(")
      assert products.(with_checkpoint) > products.(plain)
    end
  end
end
