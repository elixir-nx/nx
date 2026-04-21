defmodule Torchx.NxBlockTest do
  @moduledoc """
  Numerical coverage for `Nx.block/4`-backed APIs on Torchx.

  `Nx.fft2/2`, `Nx.ifft2/2`, `Nx.rfft/2`, and `Nx.irfft/2` route through block structs.
  `Torchx.NxDoctestTest` excludes their doctests (BinaryBackend `inspect` strings vs
  LibTorch signed zeros). Here we assert agreement with a `Nx.BinaryBackend` reference.
  """

  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  defp same_as_binary(fun_torchx, fun_binary)
       when is_function(fun_torchx, 0) and is_function(fun_binary, 0) do
    t_torchx = fun_torchx.()
    t_binary = fun_binary.()
    ref = t_binary |> Nx.backend_transfer(Torchx.Backend)
    assert Nx.all_close(t_torchx, ref)
  end

  describe "rfft / irfft (block-backed)" do
    test "rfft basic" do
      same_as_binary(
        fn -> Nx.rfft(Nx.tensor([1.0, 1.0, 0.0, 0.0])) end,
        fn -> Nx.rfft(Nx.tensor([1.0, 1.0, 0.0, 0.0], backend: Nx.BinaryBackend)) end
      )
    end

    test "rfft signed-zero case" do
      same_as_binary(
        fn -> Nx.rfft(Nx.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])) end,
        fn ->
          Nx.rfft(Nx.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], backend: Nx.BinaryBackend))
        end
      )
    end

    test "rfft with axis and length options" do
      same_as_binary(
        fn ->
          tensor = Nx.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
          Nx.rfft(tensor, axis: -2)
        end,
        fn ->
          tensor =
            Nx.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], backend: Nx.BinaryBackend)

          Nx.rfft(tensor, axis: -2)
        end
      )
    end

    test "rfft vectorized" do
      same_as_binary(
        fn ->
          Nx.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
          |> Nx.vectorize(:x)
          |> Nx.rfft()
        end,
        fn ->
          Nx.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], backend: Nx.BinaryBackend)
          |> Nx.vectorize(:x)
          |> Nx.rfft()
        end
      )
    end

    test "irfft basic" do
      same_as_binary(
        fn -> Nx.irfft(Nx.tensor([2.0, Complex.new(1.0, -1.0), 0.0])) end,
        fn ->
          Nx.irfft(Nx.tensor([2.0, Complex.new(1.0, -1.0), 0.0], backend: Nx.BinaryBackend))
        end
      )
    end

    test "irfft with length" do
      same_as_binary(
        fn -> Nx.irfft(Nx.tensor([5.0, 1.0, -1.0, 1.0])) end,
        fn -> Nx.irfft(Nx.tensor([5.0, 1.0, -1.0, 1.0], backend: Nx.BinaryBackend)) end
      )
    end

    test "irfft vectorized" do
      same_as_binary(
        fn ->
          Nx.tensor([[2.0, Complex.new(1.0, -1.0), 0.0], [4.0, 0.0, 0.0]])
          |> Nx.vectorize(:x)
          |> Nx.irfft()
        end,
        fn ->
          Nx.tensor([[2.0, Complex.new(1.0, -1.0), 0.0], [4.0, 0.0, 0.0]],
            backend: Nx.BinaryBackend
          )
          |> Nx.vectorize(:x)
          |> Nx.irfft()
        end
      )
    end
  end

  describe "fft2 / ifft2 (block-backed)" do
    test "fft2 simple matrix" do
      same_as_binary(
        fn ->
          Nx.tensor([[1, 0, 1, 0], [1, 1, 1, 1]]) |> Nx.fft2()
        end,
        fn ->
          Nx.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], backend: Nx.BinaryBackend)
          |> Nx.fft2()
        end
      )
    end

    test "ifft2 with power_of_two lengths" do
      same_as_binary(
        fn ->
          Nx.ifft2(
            Nx.tensor([[16, 16, 0], [16, 16, 0], [16, 16, -16]]),
            lengths: [:power_of_two, :power_of_two]
          )
        end,
        fn ->
          Nx.ifft2(
            Nx.tensor([[16, 16, 0], [16, 16, 0], [16, 16, -16]], backend: Nx.BinaryBackend),
            lengths: [:power_of_two, :power_of_two]
          )
        end
      )
    end

    test "fft2 with axes and lengths" do
      same_as_binary(
        fn ->
          Nx.fft2(
            Nx.tensor([
              [[[1, 0, 1, 0, 10, 10], [1, 1, 1, 1, 10, 10]]],
              [[[-2, -2, -2, -2, 20, 20], [0, 0, 0, 1, -20, -20]]]
            ]),
            lengths: [2, 4]
          )
        end,
        fn ->
          Nx.tensor(
            [
              [[[1, 0, 1, 0, 10, 10], [1, 1, 1, 1, 10, 10]]],
              [[[-2, -2, -2, -2, 20, 20], [0, 0, 0, 1, -20, -20]]]
            ],
            backend: Nx.BinaryBackend
          )
          |> Nx.fft2(lengths: [2, 4])
        end
      )
    end

    test "ifft2 vectorized" do
      same_as_binary(
        fn ->
          tensor =
            Nx.tensor([
              [[[8, 0, 8, 0, 10, 10], [8, 8, 8, 8, 10, 10]]],
              [[[-16, -16, -16, -16, 20, 20], [0, 0, 0, 8, -20, -20]]]
            ])
            |> Nx.vectorize(:x)

          Nx.ifft2(tensor, lengths: [2, 4])
        end,
        fn ->
          tensor =
            Nx.tensor(
              [
                [[[8, 0, 8, 0, 10, 10], [8, 8, 8, 8, 10, 10]]],
                [[[-16, -16, -16, -16, 20, 20], [0, 0, 0, 8, -20, -20]]]
              ],
              backend: Nx.BinaryBackend
            )
            |> Nx.vectorize(:x)

          Nx.ifft2(tensor, lengths: [2, 4])
        end
      )
    end
  end
end
