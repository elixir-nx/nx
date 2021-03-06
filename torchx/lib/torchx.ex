defmodule Torchx do
  @moduledoc """

  Torchx behaviour that is different from BinaryBackend:

  1. Torchx doesn't support u16/u32/u64. Only u8 is supported.

      iex> Nx.tensor([1, 2, 3], type: {:u, 16}, backend: Torchx.Backend)
      ** (ArgumentError) Torchx does not support unsigned 16 bit integer


  2. Torchx doesn't support u8 on sums, you should convert input to signed integer.

      iex> Nx.sum(Nx.tensor([1, 2, 3], type: {:u, 8}, backend: Torchx.Backend))
      ** (ArgumentError) Torchx does not support unsigned 64 bit integer (explicitly cast the input tensor to a signed integer before taking sum)

  3. Torchx rounds half-to-even, while Elixir rounds half-away-from-zero.
     So, in Elixir round(0.5) == 1.0, while in Torchx round(0.5) == 0.0.

      iex> Nx.tensor([-1.5, -0.5, 0.5, 1.5], backend: Torchx.Backend) |> Nx.round()
      #Nx.Tensor<
        f32[4]
        [-2.0, 0.0, 0.0, 2.0]
      >

    While binary backend will do:

      iex> Nx.tensor([-1.5, -0.5, 0.5, 1.5], backend: Nx.BinaryBackend) |> Nx.round()
      #Nx.Tensor<
        f32[4]
        [-2.0, -1.0, 1.0, 2.0]
      >

  """
end
