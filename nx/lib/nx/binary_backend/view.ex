defmodule Nx.BinaryBackend.View do
  alias Nx.BinaryBackend.View
  alias Nx.BinaryBackend.WeightedShape
  alias Nx.BinaryBackend.Traverser

  defstruct size: nil,
            weighted_shape: nil,
            weight: 1,
            changes: [],
            must_be_resolved?: false,
            reversed?: false

  def build(shape) do
    %View{
      size: Nx.size(shape),
      weighted_shape: WeightedShape.build(shape)
    }
  end

  def size(%View{size: s}), do: s

  def must_be_resolved?(%View{must_be_resolved?: r}), do: r

  def has_changes?(%View{changes: c}), do: c != []

  def meta_aggregate(view, axes) do
    %View{view | must_be_resolved?: true}
    |> add_change(:aggregate, axes)
    |> update_weighted_shape(fn ws -> WeightedShape.aggregate(ws, axes) end)
  end

  def meta_transpose(view, axes) do
    view
    |> add_change(:transpose, axes)
    |> update_weighted_shape(fn ws -> WeightedShape.transpose(ws, axes) end)
  end

  def meta_reverse(view) do
    view = add_change(view, :reverse, true)
    %View{view | reversed?: !view.reversed?}
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
    view
    |> add_change(:type, type)
    |> update_weighted_shape(fn ws -> WeightedShape.with_weight(ws, weight) end)
  end

  defp add_change(%View{changes: changes} = view, key, value) do
    %View{view | changes: [{key, value} | changes]}
  end

  defp update_weighted_shape(%View{weighted_shape: ws} = view, fun) do
    %View{view | weighted_shape: fun.(ws)}
  end

  def build_traverser(%View{} = view) do
    %View{
      size: size,
      weighted_shape: weighted_shape,
      weight: weight,
      reversed?: reversed?
    } = view

    {offsets_ws, readers_ws} =
      case weighted_shape do
        {_, _} = offsets_and_readers ->
          offsets_and_readers

        offsets when is_list(offsets) ->
          WeightedShape.aggregate(offsets, [])
      end

    Traverser.build_from_parts(size, offsets_ws, readers_ws, weight, reversed?)
  end
end
