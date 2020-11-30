defmodule Exla.Buffer do
  @moduledoc """
  An EXLA Buffer.

  An EXLA Buffer is the data passed as input and retrieved as output
  to/from EXLA Executables. An EXLA buffer is one of:

    1) A Binary
    2) A reference to an `xla::ScopedShapedBuffer`

  An `xla::ScopedShapedBuffer` is an "owning" wrapper around an
  `xla::ShapedBuffer`. Shaped Buffers are just buffers of data
  with an underlying XLA shape. Scoped Shaped Buffers are said
  to be an "owning" wrapper because they represent an allocated
  portion of memory on a specified device (like a GPU) owned by
  that device. Device memory is allocated upon creation of the
  `xla::ScopedShapedBuffer` and deallocated upon its destruction.
  """

  alias __MODULE__
  alias Exla.{Client, Shape}

  @enforce_keys [:shape]
  defstruct [:data, :ref, :shape]

  @doc """
  Creates a new buffer.

  The argument is either a `binary`, which won't be placed on the
  device unless `place_on_device` is called, or a `buffer.ref` from
  a previous buffer.
  """
  def buffer(binary_or_reference_pair, shape)

  def buffer({reference, client_name}, shape = %Shape{}) when is_reference(reference) do
    %Buffer{data: nil, ref: {reference, client_name}, shape: shape}
  end

  def buffer(binary, shape = %Shape{}) when is_bitstring(binary) do
    %Buffer{data: binary, ref: nil, shape: shape}
  end

  @doc """
  Places the given `buffer` on the given `device` using `client`.
  """
  def place_on_device(buffer = %Buffer{}, client = %Client{}, ordinal) when is_integer(ordinal) do
    ordinal = Client.check_device_compatibility!(client, ordinal)

    ref =
      Exla.NIF.binary_to_device_mem(client.ref, buffer.data, buffer.shape.ref, ordinal)
      |> unwrap!()

    %Buffer{buffer | data: nil, ref: {ref, client.name}}
  end

  @doc """
  Reads the underlying buffer ref.

  This copies the underlying device memory into a binary without destroying it.
  """
  def read({ref, client_name}) do
    client = Exla.Client.fetch!(client_name)
    binary = Exla.NIF.read_device_mem(client.ref, ref) |> unwrap!()
    binary
  end

  @doc """
  Deallocates underlying buffer ref.

  Returns `:ok` | `:already_deallocated`.
  """
  def deallocate({ref, _}) do
    Exla.NIF.deallocate_device_mem(ref) |> unwrap!()
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
