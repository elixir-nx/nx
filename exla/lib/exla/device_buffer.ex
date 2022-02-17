defmodule EXLA.DeviceDeviceBuffer do
  @moduledoc """
  An EXLA DeviceDeviceBuffer for data allocated in the device.
  """

  alias __MODULE__
  alias EXLA.{Client, Shape}

  @enforce_keys [:ref, :client_name, :device_id, :shape]
  defstruct [:ref, :client_name, :device_id, :shape]

  @doc false
  def from_ref(ref, %Client{name: name}, device_id, shape) when is_reference(ref) do
    %DeviceDeviceBuffer{ref: ref, client_name: name, device_id: device_id, shape: shape}
  end

  @doc """
  Places the given binary `buffer` on the given `device` using `client`.
  """
  def place_on_device(data, %Shape{} = shape, client = %Client{}, device_id)
      when is_integer(device_id) and is_binary(data) do
    ref =
      client.ref
      |> EXLA.NIF.binary_to_device_mem(data, shape.ref, device_id)
      |> unwrap!()

    %DeviceDeviceBuffer{ref: ref, client_name: client.name, device_id: device_id, shape: shape}
  end

  @doc """
  Reads `size` from the underlying buffer ref.

  This copies the underlying device memory into a binary
  without destroying it. If `size` is negative, then it
  reads the whole buffer.
  """
  def read(%DeviceDeviceBuffer{ref: ref, client_name: client_name}, size \\ -1) do
    client = EXLA.Client.fetch!(client_name)
    binary = EXLA.NIF.read_device_mem(client.ref, ref, size) |> unwrap!()
    binary
  end

  @doc """
  Deallocates the underlying buffer.

  Returns `:ok` | `:already_deallocated`.
  """
  def deallocate(%DeviceDeviceBuffer{ref: ref}),
    do: EXLA.NIF.deallocate_device_mem(ref) |> unwrap!()

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
