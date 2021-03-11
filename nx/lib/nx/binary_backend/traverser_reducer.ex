defmodule Nx.BinaryBackend.TraverserReducer do
  alias Nx.BinaryBackend
  alias Nx.BinaryBackend.Bits
  alias Nx.BinaryBackend.Traverser
  alias Nx.Tensor, as: T

  def bin_reduce(out, tensor, acc, opts, fun) do
    %T{type: {_, sizeof}, shape: shape} = tensor
    %T{type: type_out} = out

    axes = Keyword.get(opts, :axes, []) || []
    data = BinaryBackend.to_binary(tensor)

    trav =
      if axes == [] do
        Traverser.range(Nx.size(shape))
      else
        Traverser.build(shape, aggregate: axes)
      end

    Traverser.reduce_aggregates(trav, [], fn aggs, outer_acc ->
      {agg_out, _} =
        Enum.reduce(aggs, {[], acc}, fn o, {_, agg_acc} ->
          offset = o * sizeof
          <<_::size(offset)-bitstring, bin::size(sizeof)-bitstring, _::bitstring>> = data
          fun.(bin, agg_acc)
        end)

      [outer_acc, Bits.from_scalar(agg_out, type_out)]
    end)
  end

  def bin_zip_reduce(out, t1, axes1, t2, axes2, acc, fun) do
    %T{type: type_out} = out
    %T{type: {_, sizeof1}, shape: shape1} = t1
    %T{type: {_, sizeof2}, shape: shape2} = t2

    trav1 = Traverser.build(shape1, aggregate: axes1, weight: sizeof1)
    trav2 = Traverser.build(shape2, aggregate: axes2, weight: sizeof2)

    data1 = BinaryBackend.to_binary(t1)
    data2 = BinaryBackend.to_binary(t2)

    Traverser.zip_reduce_aggregates(
      trav1,
      trav2,
      [],
      acc,
      fn o1, o2, acc2 ->
        <<_::size(o1)-bitstring, bin1::size(sizeof1)-bitstring, _::bitstring>> = data1
        <<_::size(o2)-bitstring, bin2::size(sizeof2)-bitstring, _::bitstring>> = data2
        {_, acc3} = fun.(bin1, bin2, acc2)
        acc3
      end,
      fn acc3, outer_acc ->
        [outer_acc, Bits.from_scalar(acc3, type_out)]
      end
    )
  end
end
