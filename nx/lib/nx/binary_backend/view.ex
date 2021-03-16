defmodule Nx.BinaryBackend.View do
  alias Nx.BinaryBackend.View
  alias Nx.BinaryBackend.WeightedShape
  alias Nx.BinaryBackend.Traverser
  # alias Nx.BinaryBackend.Bits

  import Nx.Shared

  defstruct weighted_shape: nil,
            type: nil,
            changed?: false

  def build(shape) do
    %View{
      weighted_shape: WeightedShape.build(shape)
    }
  end

  def shape_with_aggregates(%View{weighted_shape: ws}) do
    WeightedShape.shape_with_aggregates(ws)
  end

  def shape_without_aggregates(%View{weighted_shape: ws}) do
    WeightedShape.shape_without_aggregates(ws)
  end

  def must_be_resolved?(%View{weighted_shape: {_, _, _}}), do: true
  def must_be_resolved?(_), do: false

  def has_changes?(%View{changed?: c}), do: c

  def aggregate(view, axes) do
    change(view, fn ws -> WeightedShape.aggregate(ws, axes) end)
  end

  def transpose(view, axes) do
    change(view, fn ws -> WeightedShape.transpose(ws, axes) end)
  end

  def reverse(view, axes) do
    change(view, fn ws -> WeightedShape.reverse(ws, axes) end)
  end

  def dilate(view, dilation) do
    change(view, fn ws -> WeightedShape.dilate(ws, dilation) end)
  end

  def limit(view, limits) do
    change(view, fn ws -> WeightedShape.limit(ws, limits) end)
  end

  def with_type(view, type) do
    %View{view | type: type}
  end

  defp change(%View{weighted_shape: ws} = view, fun) when is_list(ws) do
    %View{view | weighted_shape: fun.(ws), changed?: true}
  end

  def build_traverser(%View{weighted_shape: ws, type: {_, _} = type}) do
    Traverser.build(ws, type)
  end

  def build_traverser(%View{weighted_shape: ws, type: nil}) do
    Traverser.build(ws)
  end

  def map_aggregates(view, type, data, mapper) do
    match_types [type] do
      view
      |> with_type(type)
      |> build_traverser()
      |> Traverser.reduce_aggregates([], fn {agg_o, agg_indices}, acc ->
        out =
          agg_indices
          |> Enum.map(fn agg_i ->
            <<
              _::size(agg_o)-bitstring,
              _::size(agg_i)-bitstring,
              match!(bin_num, 0),
              _::bitstring
            >> = data
            read!(bin_num, 0)
          end)
          |> mapper.()
          |> Enum.map(fn n ->
            <<write!(n, 0)>>
          end)
        [acc | out]
      end)
    end
  end
end
