defmodule Exla.Builder do
  alias __MODULE__, as: Builder
  alias Exla.Op
  alias Exla.Computation
  @enforce_keys [:ref]
  defstruct [:ref]

  def new(name) do
    case Exla.NIF.new_builder(name) do
      {:ok, ref} -> {:ok, %Builder{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  def build(root = %Op{}) do
    case Exla.NIF.build(root.builder, root.ref) do
      {:ok, ref} -> {:ok, %Computation{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end
end
