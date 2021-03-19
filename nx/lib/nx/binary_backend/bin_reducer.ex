defmodule Nx.BinaryBackend.BinReducer do
  import Nx.Shared
  alias Nx.BinaryBackend
  alias Nx.BinaryBackend.Bits

  alias Nx.Tensor, as: T

  def bin_reduce(out, tensor, acc, opts, fun) do
    %T{type: {_, size}, shape: shape} = tensor

    view =
      if axes = opts[:axes] do
        BinaryBackend.aggregate_axes(BinaryBackend.to_binary(tensor), axes, shape, size)
      else
        [BinaryBackend.to_binary(tensor)]
      end

    for axis <- view do
      {result, _} =
        for <<bin::size(size)-bitstring <- axis>>, reduce: {<<>>, acc} do
          {_, acc} -> fun.(bin, acc)
        end

      Bits.from_number(result, out.type)
    end
  end

  def bin_zip_reduce(%{type: type}, t1, [], t2, [], acc, fun) do
    %{type: {_, s1}} = t1
    %{type: {_, s2}} = t2
    b1 = BinaryBackend.to_binary(t1)
    b2 = BinaryBackend.to_binary(t2)

    match_types [t1.type, t2.type] do
      for <<d1::size(s1)-bitstring <- b1>>, <<d2::size(s2)-bitstring <- b2>>, into: <<>> do
        {result, _} = fun.(d1, d2, acc)
        Bits.from_number(result, type)
      end
    end
  end

  def bin_zip_reduce(%{type: type}, t1, [_ | _] = axes1, t2, [_ | _] = axes2, acc, fun) do
    {_, s1} = t1.type
    {_, s2} = t2.type

    v1 = BinaryBackend.aggregate_axes(BinaryBackend.to_binary(t1), axes1, t1.shape, s1)
    v2 = BinaryBackend.aggregate_axes(BinaryBackend.to_binary(t2), axes2, t2.shape, s2)

    for b1 <- v1, b2 <- v2 do
      {num, _acc} = bin_zip_reduce_axis(b1, b2, s1, s2, <<>>, acc, fun)
      Bits.from_number(num, type)
    end
  end

  # Helper for reducing down a single axis over two tensors,
  # returning tensor data and a final accumulator.
  defp bin_zip_reduce_axis(<<>>, <<>>, _s1, _s2, bin, acc, _fun),
    do: {bin, acc}

  defp bin_zip_reduce_axis(b1, b2, s1, s2, _bin, acc, fun) do
    <<x::size(s1)-bitstring, rest1::bitstring>> = b1
    <<y::size(s2)-bitstring, rest2::bitstring>> = b2
    {bin, acc} = fun.(x, y, acc)
    bin_zip_reduce_axis(rest1, rest2, s1, s2, bin, acc, fun)
  end
end
