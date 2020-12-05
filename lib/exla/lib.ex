defmodule Exla.Lib do
  @moduledoc """
  High-level operations.
  """

  alias Exla.{Builder, Op, Shape}

  @doc """
  Computes the sum of the given operation.  
  """
  def sum(%Builder{} = builder, %Op{} = op) do
    op_shape = Op.get_shape(op)
    reduction_shape = Shape.make_shape(op_shape.dtype, {})

    unique = System.unique_integer([:positive])
    sub_builder = Builder.new(builder, builder.name <> "-sum-" <> Integer.to_string(unique))
    a = Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Op.parameter(sub_builder, 1, reduction_shape, "b")
    add = Op.add(a, b)
    reduction = Builder.build(add)

    init_value = Op.constant_r0(builder, 0, reduction_shape.dtype)
    Op.reduce(op, init_value, reduction, all_dimensions(op_shape.dims))
  end

  # Converts {3, 255, 255} into {0, 1, 2}
  defp all_dimensions(dims), do: List.to_tuple(all_dimensions(0, tuple_size(dims)))
  defp all_dimensions(i, n) when i < n, do: [i | all_dimensions(i + 1, n)]
  defp all_dimensions(_, _), do: []
end
