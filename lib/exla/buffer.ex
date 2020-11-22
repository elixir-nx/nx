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

  defstruct [:data, :ref, :shape]

  def to_bitstring(%Client{ref: client}, %Buffer{data: nil, ref: ref}) do
    {:ok, binary} = Exla.NIF.shaped_buffer_to_binary(client, ref)
    binary
  end

  def to_bitstring(_, %Buffer{data: data}), do: data

  def to_shaped_buffer(%Client{ref: client}, %Buffer{data: data, ref: nil, shape: shape}) do
    {:ok, buffer} = Exla.NIF.binary_to_shaped_buffer(client, data, shape.ref, 0)
    buffer
  end

  def to_shaped_buffer(_, %Buffer{ref: ref}), do: ref

end
