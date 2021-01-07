defmodule Nx.Device do
  @moduledoc """
  Defines a device where the tensor is allocated.
  """

  @type state :: term

  @doc """
  Allocates the given data on the device.
  """
  @callback allocate(binary, Nx.Tensor.type(), Nx.Tensor.shape(), opts :: keyword) ::
              {module, state}

  @doc """
  Reads the data on the device.

  If reading a deallocated device, you must raise.
  """
  @callback read(state) :: binary

  @doc """
  Deallocates the data on the device.

  If the device was already been deallocated, returns `:already_deallocated`.
  """
  @callback deallocate(state) :: :ok | :already_deallocated
end
