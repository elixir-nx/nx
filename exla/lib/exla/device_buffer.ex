defmodule EXLA.DeviceBuffer do
  @moduledoc """
  An EXLA DeviceBuffer for data allocated in the device.
  """

  alias __MODULE__
  alias EXLA.Client

  @enforce_keys [:ref, :client_name, :device_id, :typespec]
  defstruct [:ref, :client_name, :device_id, :typespec]

  @doc false
  def from_ref(ref, %Client{name: name}, device_id, typespec) when is_reference(ref) do
    %DeviceBuffer{ref: ref, client_name: name, device_id: device_id, typespec: typespec}
  end

  @doc """
  Places the given binary `data` on the given `device` using `client`.
  """
  def place_on_device(data, %EXLA.Typespec{} = typespec, client = %Client{}, device_id)
      when is_integer(device_id) and is_bitstring(data) do
    # At the moment XLA does not support allocating a packed buffer,
    # so we unpack subbyte elements into their own bytes
    data =
      case typespec.type do
        {:u, size} when size in [2, 4] ->
          for <<x::native-size(size) <- data>>, into: <<>>, do: <<x::native-8>>

        {:s, size} when size in [2, 4] ->
          for <<x::native-signed-size(size) <- data>>, into: <<>>, do: <<x::native-signed-8>>

        _ ->
          data
      end

    ref =
      client.ref
      |> EXLA.NIF.binary_to_device_mem(data, EXLA.Typespec.nif_encode(typespec), device_id)
      |> unwrap!()

    %DeviceBuffer{ref: ref, client_name: client.name, device_id: device_id, typespec: typespec}
  end

  @doc """
  Copies buffer to device with given device ID.
  """
  def copy_to_device(
        %DeviceBuffer{ref: buffer, typespec: typespec},
        %Client{} = client,
        device_id
      )
      when is_integer(device_id) do
    ref = client.ref |> EXLA.NIF.copy_buffer_to_device(buffer, device_id) |> unwrap!()
    %DeviceBuffer{ref: ref, client_name: client.name, device_id: device_id, typespec: typespec}
  end

  @doc """
  Reads `size` from the underlying buffer ref.

  This copies the underlying device memory into a binary
  without destroying it. If `size` is negative, then it
  reads the whole buffer.
  """
  def read(%DeviceBuffer{ref: ref, typespec: typespec}, size \\ -1) do
    data = EXLA.NIF.read_device_mem(ref, size) |> unwrap!()

    # At the moment XLA does not support reading a packed buffer,
    # so we pack the elements ourselves
    case typespec.type do
      {:u, size} when size in [2, 4] ->
        for <<x::native-8 <- data>>, into: <<>>, do: <<x::native-size(size)>>

      {:s, size} when size in [2, 4] ->
        for <<x::native-signed-8 <- data>>, into: <<>>, do: <<x::native-signed-size(size)>>

      _ ->
        data
    end
  end

  @doc """
  Deallocates the underlying buffer.

  Returns `:ok` | `:already_deallocated`.
  """
  def deallocate(%DeviceBuffer{ref: ref}),
    do: EXLA.NIF.deallocate_device_mem(ref) |> unwrap!()

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
