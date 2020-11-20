defmodule Exla.Builder do
  alias __MODULE__, as: Builder
  alias Exla.Op
  alias Exla.Computation
  @enforce_keys [:ref]
  defstruct [:ref, :parent]

  def new(name) do
    {:ok, ref} = Exla.NIF.new_builder(name)
    %Builder{ref: ref, parent: nil}
  end

  def new(builder = %Builder{ref: ref}, name) do
    {:ok, ref} = Exla.NIF.create_sub_builder(ref, name)
    %Builder{ref: ref, parent: builder}
  end

  def build(root = %Op{}) do
    {:ok, ref} = Exla.NIF.build(root.builder, root.ref)
    %Computation{ref: ref}
  end
end
