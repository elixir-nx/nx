defmodule Exla.Computation do
  alias __MODULE__, as: Computation
  alias Exla.Shape

  @enforce_keys [:ref]
  defstruct [:ref]

  def get_program_shape(%Computation{ref: ref}) do
    {:ok, {input_shapes, {output_dims, output_type_str, output_ref}}} = Exla.NIF.get_program_shape(ref)

    input_shapes =
      input_shapes
      |> Enum.map(fn {dims, type, ref} -> %Shape{dims: dims, dtype: Shape.str_to_dtype(type), ref: ref} end)

    {input_shapes, %Shape{dims: output_dims, dtype: Shape.str_to_dtype(output_type_str), ref: output_ref}}
  end
end
