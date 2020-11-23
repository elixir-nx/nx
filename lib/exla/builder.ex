defmodule Exla.Builder do
  alias __MODULE__, as: Builder
  alias Exla.Op
  alias Exla.Computation
  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new(name) when is_binary(name) do
    {:ok, ref} = Exla.NIF.new_builder(name)
    %Builder{ref: ref, parent: nil, name: name}
  end

  def new(builder = %Builder{ref: ref}, name) when is_binary(name) do
    {:ok, ref} = Exla.NIF.create_sub_builder(ref, name)
    %Builder{ref: ref, parent: builder, name: name}
  end

  def build(root = %Op{}) do
    {:ok, ref} = Exla.NIF.build(root.builder, root.ref)
    %Computation{ref: ref}
  end
end
