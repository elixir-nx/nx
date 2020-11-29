defmodule Exla.Buffer do
  @moduledoc """
  An EXLA Buffer.

  An EXLA Buffer is the data passed as input and retrieved as output to/from EXLA Executables. An EXLA
  Buffer consists of two things:

    1) A Binary
    2) A reference to an `xla::ScopedShapedBuffer`

  An `xla::ScopedShapedBuffer` is an "owning" wrapper around an `xla::ShapedBuffer`. Shaped Buffers are just
  buffers of data with an underlying XLA shape. Scoped Shaped Buffers are said to be an "owning" wrapper because
  they represent an allocated portion of memory on a specified device (like a GPU) owned by that device. Device
  memory is allocated upon creation of the `xla::ScopedShapedBuffer` and deallocated upon it's destruction.
  """

  alias __MODULE__
  alias Exla.{Client, Shape}

  defstruct [:data, :ref, :shape]

  @doc """
  Create a new buffer from `binary` with given `shape`.

  The buffer will not be placed on a device until passed to `place_on_device`.
  """
  def buffer(binary, shape = %Shape{}) when is_bitstring(binary) do
    %Buffer{data: binary, ref: nil, shape: shape}
  end

  def buffer(tuple, shape = %Shape{dtype: {:t, _}}) when is_tuple(tuple) do
    %Buffer{data: tuple, ref: nil, shape: shape}
  end

  @doc """
  Create a new buffer from `binary` and place on `device` using `client`.

  If the device is a GPU, the entire binary will be consumed and the data field will be `nil`. On CPU,
  we retain a copy of the binary to ensure it is not prematurely garbage collected.
  """
  def buffer(binary, shape = %Shape{}, client = %Client{}, device = {_platform, _ordinal})
      when is_bitstring(binary) do
    {:ok, buffer = %Buffer{}} =
      place_on_device(client, %Buffer{data: binary, shape: shape}, device)

    buffer
  end

  @doc """
  Places the given `buffer` on the given `device` using `client`.
  """
  def place_on_device(client = %Client{}, buffer = %Buffer{data: data}, device = {_platform, _ordinal})
      when is_tuple(data) do
    {:ok, {platform, ordinal}} = Client.check_device_compatibility(client, device)
    ref = Exla.NIF.tuple_to_device_mem(client.ref, data, buffer.shape.ref, ordinal) |> unwrap!()
    %Buffer{buffer | data: nil, ref: {ref, platform, ordinal}}
  end

  def place_on_device(client = %Client{}, buffer = %Buffer{}, device = {_platform, _ordinal}) do
    {:ok, {platform, ordinal}} = Client.check_device_compatibility(client, device)
    ref = Exla.NIF.binary_to_device_mem(client.ref, buffer.data, buffer.shape.ref, ordinal) |> unwrap!()
    %Buffer{buffer | data: nil, ref: {ref, platform, ordinal}}
  end

  @doc """
  Reads the underlying device buffer.

  This copies the underlying device memory into a binary without destroying it.
  """
  def read(client = %Client{}, buffer = %Buffer{ref: {ref, platform, ordinal}}) do
    {:ok, _} = Client.check_device_compatibility(client, {platform, ordinal})
    binary = Exla.NIF.read_device_mem(client.ref, ref) |> unwrap!()
    binary
  end

  @doc """
  Deallocates underlying device buffer.

  Returns `:ok` | `:already_deallocated`.
  """
  def deallocate(%Buffer{ref: {ref, _, _}}) do
    Exla.NIF.deallocate_device_mem(ref) |> unwrap!()
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
