defmodule Exla.Builder do
  alias __MODULE__, as: Builder
  alias Exla.Op
  alias Exla.Computation
  @enforce_keys [:ref]
  defstruct [:ref]

  def new(name) do
    {:ok, ref} = Exla.NIF.new_builder(name)
    %Builder{ref: ref}
  end

  def build(root = %Op{}) do
    {:ok, ref} = Exla.NIF.build(root.builder, root.ref)
    %Computation{ref: ref}
  end
end
