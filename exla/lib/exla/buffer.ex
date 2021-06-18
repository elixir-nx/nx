defmodule EXLA.Buffer do
  @moduledoc """
  An EXLA Buffer.

  An EXLA Buffer is the data passed as input and retrieved as output
  to/from EXLA Executables.
  """

  alias __MODULE__
  alias EXLA.{Client, Shape}

  @enforce_keys [:shape]
  defstruct [:data, :ref, :shape]

  @doc false
  def buffer(binary_or_reference_pair, shape)

  def buffer({reference, client_name}, shape = %Shape{}) when is_reference(reference) do
    %Buffer{data: nil, ref: {reference, client_name}, shape: shape}
  end

  def buffer(binary, shape = %Shape{}) when is_binary(binary) do
    %Buffer{data: binary, ref: nil, shape: shape}
  end

  @doc """
  Places the given `buffer` on the given `device` using `client`.
  """
  def place_on_device(buffer = %Buffer{}, client = %Client{}, device_id) when is_integer(device_id) do
    ref =
      EXLA.NIF.binary_to_device_mem(client.ref, buffer.data, buffer.shape.ref, device_id)
      |> unwrap!()

    %Buffer{buffer | data: nil, ref: {ref, client.name}}
  end

  @doc """
  Reads the underlying buffer ref.

  This copies the underlying device memory into a binary without destroying it.
  """
  def read({ref, client_name}) do
    client = EXLA.Client.fetch!(client_name)
    binary = EXLA.NIF.read_device_mem(client.ref, ref) |> unwrap!()
    binary
  end

  @doc """
  Deallocates underlying buffer ref.

  Returns `:ok` | `:already_deallocated`.
  """
  def deallocate({ref, _}) do
    EXLA.NIF.deallocate_device_mem(ref) |> unwrap!()
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
  defp unwrap!(status) when is_atom(status), do: status
end
