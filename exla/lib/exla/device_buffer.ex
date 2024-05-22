defmodule EXLA.DeviceBuffer do
  @moduledoc """
  An EXLA DeviceBuffer for data allocated in the device.
  """

  alias __MODULE__
  alias EXLA.Client

  @enforce_keys [:ref, :client_name, :device_id, :typespec]
  defstruct [:ref, :client_name, :device_id, :typespec]

  @doc false
  def from_ref(ref, :iree, device_id, typespec) when is_reference(ref) do
    %DeviceBuffer{ref: ref, client_name: :iree, device_id: device_id, typespec: typespec}
  end

  def from_ref(ref, %Client{name: name}, device_id, typespec) when is_reference(ref) do
    %DeviceBuffer{ref: ref, client_name: name, device_id: device_id, typespec: typespec}
  end

  @doc """
  Places the given binary `buffer` on the given `device` using `client`.
  """
  def place_on_device(data, %EXLA.Typespec{} = typespec, client = %Client{}, device_id)
      when is_integer(device_id) and is_binary(data) do
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
  def read(buffer, size \\ -1)

  @downcast_types [f: 64, c: 128]

  def read(%DeviceBuffer{typespec: typespec, ref: ref, client_name: :iree}, size) do
    {s, w} = typespec.type

    size =
      if size == -1 do
        div(w, 8) * Tuple.product(typespec.shape)
      else
        size
      end

    read_size =
      if {s, w} in @downcast_types do
        div(size, 2)
      else
        size
      end

    data = EXLA.MLIR.IREE.read(ref, read_size) |> unwrap!()

    if read_size != size do
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        data |> Nx.from_binary({s, div(w, 2)}) |> Nx.as_type({s, w}) |> Nx.to_binary()
      end)
    else
      data
    end
  end

  def read(%DeviceBuffer{ref: ref}, size) do
    EXLA.NIF.read_device_mem(ref, size) |> unwrap!()
  end

  @doc """
  Deallocates the underlying buffer.

  Returns `:ok` | `:already_deallocated`.
  """
  def deallocate(%DeviceBuffer{ref: ref, client_name: :iree}),
    do: EXLA.MLIR.IREE.deallocate_buffer(ref) |> unwrap!()

  def deallocate(%DeviceBuffer{ref: ref}),
    do: EXLA.NIF.deallocate_device_mem(ref) |> unwrap!()

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
