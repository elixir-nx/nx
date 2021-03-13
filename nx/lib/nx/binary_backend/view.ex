defmodule Nx.BinaryBackend.View do
  alias Nx.BinaryBackend.View
  alias Nx.BinaryBackend.WeightedShape
  alias Nx.BinaryBackend.Traverser

  defstruct size: nil,
            weighted_shape: nil,
            weight: 1,
            changes: [],
            must_be_resolved?: false

  def build(shape) do
    %View{
      size: Nx.size(shape),
      weighted_shape: WeightedShape.build(shape)
    }
  end

  def size(%View{size: s}), do: s

  def must_be_resolved?(%View{must_be_resolved?: r}), do: r

  def has_changes?(%View{changes: c}), do: c != []

  def aggregate(view, axes) do
    %View{view | must_be_resolved?: true}
    |> add_change(:aggregate, axes)
    |> update_weighted_shape(fn ws -> WeightedShape.aggregate(ws, axes) end)
  end

  def transpose(view, axes) do
    view
    |> add_change(:transpose, axes)
    |> update_weighted_shape(fn ws -> WeightedShape.transpose(ws, axes) end)
  end

  def reverse(view, axes) do
    view
    |> add_change(:reverse, axes)
    |> update_weighted_shape(fn ws -> WeightedShape.reverse(ws, axes) end)
  end

  def dilate(view, dilation) do
    view
    |> add_change(:dilate, dilation)
    |> update_weighted_shape(fn ws -> WeightedShape.dilate(ws, dilation) end)
  end

  def limit(view, limits) do
    view
    |> add_change(:limit, limits)
    |> update_weighted_shape(fn ws -> WeightedShape.limit(ws, limits) end)
  end

  def with_type(view, {_, weight} = type) do
    %View{
      view |
      weight: weight
    }
    |> add_change(:type, type)
    |> update_weighted_shape(fn
      {o, r} -> 
        ow = WeightedShape.with_weight(o, weight)
        rw = WeightedShape.with_weight(r, weight)
        {ow, rw}
      ws when is_list(ws) ->
        WeightedShape.with_weight(ws, weight)
    end)
  end

  defp add_change(%View{changes: changes} = view, key, value) do
    %View{view | changes: [{key, value} | changes]}
  end

  defp update_weighted_shape(%View{weighted_shape: ws} = view, fun) do
    %View{view | weighted_shape: fun.(ws)}
  end

  def build_traverser(%View{size: s, weight: w, weighted_shape: ws}) do
    Traverser.build(s, w, ws)
  end
end
