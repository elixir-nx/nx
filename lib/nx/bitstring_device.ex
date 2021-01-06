defmodule Nx.BitStringDevice do
  @moduledoc """
  The default implementation of `Nx.Device` that uses the VM binarys.
  """

  # TODO: Rename to binary device
  # TODO: Rename to_binary to to_binary

  @behaviour Nx.Device

  @impl true
  def allocate(data, _type, _shape, _opts), do: {__MODULE__, data}

  @impl true
  def read(data), do: data

  @impl true
  def deallocate(_data), do: :ok
end
