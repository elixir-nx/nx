defmodule Nx.BitStringDevice do
  @moduledoc """
  The default implementation of `Nx.Device` that uses the VM bitstrings.
  """

  @behaviour Nx.Device

  @impl true
  def allocate(data, _type, _shape, _opts), do: {__MODULE__, data}

  @impl true
  def read(data), do: data

  @impl true
  def deallocate(_data), do: :ok
end