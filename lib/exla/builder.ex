defmodule EXLA.Builder do
  alias __MODULE__
  alias EXLA.{Computation, Op}

  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new(name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.new_builder(name)
    %Builder{ref: ref, parent: nil, name: name}
  end

  def new(builder = %Builder{ref: ref}, name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.create_sub_builder(ref, name)
    %Builder{ref: ref, parent: builder, name: name}
  end

  def build(root = %Op{}) do
    shape = EXLA.Op.get_shape(root)
    {:ok, ref} = EXLA.NIF.build(root.builder, root.ref)
    %Computation{ref: ref, output_shape: shape}
  end
end
