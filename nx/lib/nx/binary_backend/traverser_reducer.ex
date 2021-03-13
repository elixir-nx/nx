defmodule Nx.BinaryBackend.TraverserReducer do
  alias Nx.BinaryBackend.Bits
  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.TensorView
  alias Nx.BinaryBackend.View
  alias Nx.Tensor, as: T

  def bin_reduce(out, tensor, acc, opts, fun) do
    %T{type: {_, sizeof}, shape: shape} = tensor
    %T{type: type_out} = out

    axes = Keyword.get(opts, :axes, []) || []

    tensor = TensorView.resolve_if_required(tensor)
    data = TensorView.raw_binary(tensor)

    trav =
      if axes == [] do
        Traverser.range(Nx.size(shape))
      else
        tensor
        |> TensorView.get_or_create_view()
        |> View.aggregate(axes)
        |> View.build_traverser()
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
    %T{type: {_, sizeof1} = type1} = t1
    %T{type: {_, sizeof2} = type2} = t2

    t1 = TensorView.resolve(t1)
    t2 = TensorView.resolve(t2)

    bin1 = TensorView.raw_binary(t1)
    bin2 = TensorView.raw_binary(t2)

    trav1 = 
      t1
      |> TensorView.get_or_create_view()
      |> View.with_type(type1)
      |> View.aggregate(axes1)
      |> View.build_traverser()
    
    trav2 =
      t2
      |> TensorView.get_or_create_view()
      |> View.with_type(type2)
      |> View.aggregate(axes2)
      |> View.build_traverser()

    Traverser.zip_reduce_aggregates(
      trav1,
      trav2,
      [],
      acc,
      fn o1, o2, acc2 ->
        <<_::size(o1)-bitstring, b1::size(sizeof1)-bitstring, _::bitstring>> = bin1
        <<_::size(o2)-bitstring, b2::size(sizeof2)-bitstring, _::bitstring>> = bin2
        {_, acc3} = fun.(b1, b2, acc2)
        acc3
      end,
      fn acc3, outer_acc ->
        [outer_acc, Bits.from_scalar(acc3, type_out)]
      end
    )
  end
end
