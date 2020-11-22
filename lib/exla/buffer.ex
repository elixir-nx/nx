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
  def buffer(binary, shape = %Shape{}, client = %Client{}, device = {platform, ordinal}) when is_bitstring(binary) do
    {:ok, buffer = %Buffer{}} = place_on_device(client, %Buffer{data: binary, shape: shape}, device)
    buffer
  end

  @doc """
  Places the given `buffer` on the given `device` using `client`.

  If the device is a GPU, the entire binary will be consumed and the data field will be `nil`. On CPU,
  we retain a copy of the binary to ensure it is not prematurely garbage collected.
  """
  def place_on_device(client = %Client{}, buffer = %Buffer{}, device = {:cpu, _ordinal}) do
    ref = to_shaped_buffer(client, buffer, device)
    %Buffer{buffer | ref: ref}
  end

  def place_on_device(client = %Client{}, buffer = %Buffer{}, device = {_platform, _ordinal}) do
    ref = to_shaped_buffer(client, buffer, device)
    %Buffer{data: nil, shape: buffer.shape, ref: ref}
  end

  @doc """
  Returns the buffer's underlying data as a bitstring.

  As is, this is destructive. Using the reference `ref` will lead to undefined behaviour.
  """
  def to_bitstring(%Client{ref: client}, %Buffer{data: nil, ref: ref}) do
    {:ok, binary} = Exla.NIF.shaped_buffer_to_binary(client, ref)
    binary
  end
  def to_bitstring(_, %Buffer{data: data}), do: data

  @doc """
  Returns a reference to the underlying `ShapedBuffer` on `device`.
  """
  def to_shaped_buffer(client = %Client{}, %Buffer{data: data, ref: nil, shape: shape}, device = {_platform, _ordinal}) do
    {:ok, {_platform, ordinal}} = Client.check_device_compatibility(client, device)
    {:ok, buffer} = Exla.NIF.binary_to_shaped_buffer(client.ref, data, shape.ref, ordinal)
    buffer
  end

  def to_shaped_buffer(_, %Buffer{ref: ref}, _), do: ref

end
