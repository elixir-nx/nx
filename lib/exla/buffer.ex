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

  alias __MODULE__, as: Buffer
  alias Exla.Client
  alias Exla.Shape

  defstruct [:data, :ref, :shape]

  def buffer(binary, dtype = {_type, _size}, dims) when is_bitstring(binary) do
    buffer(binary, Exla.Shape.make_shape(dtype, dims))
  end

  @doc """
  Create a new buffer from `binary` with given `shape`.

  The buffer will not be placed on a device until passed to `place_on_device`.
  """
  def buffer(binary, shape) when is_bitstring(binary) do
    %Buffer{data: binary, ref: nil, shape: shape}
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

  If the device is a GPU, the entire binary will be consumed and the data field will be `nil`. On CPU,
  we retain a copy of the binary to ensure it is not prematurely garbage collected.
  """
  def place_on_device(client = %Client{}, buffer = %Buffer{}, device = {:host, ordinal}) do
    {:ok, {_, ordinal}} = Client.check_device_compatibility(client, device)
    ref = Exla.NIF.binary_to_device_mem(client.ref, buffer.data, buffer.shape.ref, ordinal) |> unwrap!()
    %Buffer{buffer | ref: {ref, :host, ordinal}}
  end

  def place_on_device(client = %Client{}, buffer = %Buffer{}, device = {:cuda, ordinal}) do
    {:ok, {_, ordinal}} = Client.check_device_compatibility(client, device)
    ref = Exla.NIF.binary_to_device_mem(client.ref, buffer.data, buffer.shape.ref, ordinal) |> unwrap!()
    %Buffer{buffer | data: nil, ref: {ref, :cuda, ordinal}}
  end

  @doc """
  Reads the underlying device buffer.

  This copies the underlying device memory into a binary without destroying it.
  """
  def read(client = %Client{}, buffer = %Buffer{ref: {ref, :cuda, ordinal}}) do
    {:ok, _} = Client.check_device_compatibility(client, {:cuda, ordinal})
    binary = Exla.NIF.read_device_mem(client.ref, ref) |> unwrap!()
    binary
  end

  def read(client = %Client{}, buffer = %Buffer{data: data, ref: {_, :host, _}}), do: data

  def deallocate(%Buffer{ref: nil}), do: raise("Attempt to deallocate nothing.");
  def deallocate(%Buffer{ref: {ref, _, _}}) do
    Exla.NIF.deallocate_device_mem(ref) |> unwrap!()
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
