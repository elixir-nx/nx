defmodule EXLA.FFI.TermTensor do
  @moduledoc """
  Helper module for encoding BEAM terms as tensors for XLA FFI custom calls.

  This module provides utilities to convert Erlang/Elixir terms (like PIDs and tags)
  into fixed-size `u8` tensors that can be passed to XLA computations. The NIF side
  can then decode these using `enif_binary_to_term`.

  ## Encoding Strategy

  Terms are encoded using `:erlang.term_to_binary/1` and the resulting binary
  is represented as a fixed-size `u8` tensor. The sizes are determined at compile
  time based on typical term encodings.

  ## Examples

      # Encode a PID as a tensor
      pid_tensor = EXLA.FFI.TermTensor.pid_to_tensor(self())

      # Encode a tag (e.g., reference + PID) as a tensor
      tag = {make_ref(), self()}
      tag_tensor = EXLA.FFI.TermTensor.tag_to_tensor(tag)

  """

  @pid_size byte_size(:erlang.term_to_binary(self()))
  @tag_size byte_size(:erlang.term_to_binary({make_ref(), self()}))

  @type t :: Nx.Tensor.t()

  @doc """
  Returns the size in bytes needed to encode a PID.
  """
  def pid_size, do: @pid_size

  @doc """
  Returns the size in bytes needed to encode a tag.
  """
  def tag_size, do: @tag_size

  @doc """
  Converts a PID to a tensor representation.

  The resulting tensor has shape `{#{@pid_size}}` and type `:u8`, containing
  the bytes of the term_to_binary encoding of the PID.

  ## Examples

      pid_tensor = EXLA.FFI.TermTensor.pid_to_tensor(self())

  """
  @spec pid_to_tensor(pid()) :: t()
  def pid_to_tensor(pid) when is_pid(pid) do
    bin = :erlang.term_to_binary(pid)
    assert_size!(bin, @pid_size, :pid)
    bin |> Nx.from_binary(:u8) |> Nx.reshape({@pid_size})
  end

  @doc """
  Converts a tag term to a tensor representation.

  The resulting tensor has shape `{#{@tag_size}}` and type `:u8`, containing
  the bytes of the term_to_binary encoding of the tag.

  Tags are typically tuples like `{ref, pid}` used to track infeed state.

  ## Examples

      tag = {make_ref(), self()}
      tag_tensor = EXLA.FFI.TermTensor.tag_to_tensor(tag)

  """
  @spec tag_to_tensor(term()) :: t()
  def tag_to_tensor(tag) do
    bin = :erlang.term_to_binary(tag)
    assert_size!(bin, @tag_size, :tag)
    bin |> Nx.from_binary(:u8) |> Nx.reshape({@tag_size})
  end

  @doc """
  Decodes a PID from a binary (for testing purposes).

  This is the Elixir equivalent of what the NIF will do with `enif_binary_to_term`.

  ## Examples

      pid = self()
      tensor = EXLA.FFI.TermTensor.pid_to_tensor(pid)
      bin = Nx.to_binary(tensor)
      ^pid = EXLA.FFI.TermTensor.binary_to_pid(bin)

  """
  @spec binary_to_pid(binary()) :: pid()
  def binary_to_pid(bin) when byte_size(bin) == @pid_size do
    case :erlang.binary_to_term(bin) do
      pid when is_pid(pid) -> pid
      other -> raise ArgumentError, "Expected PID, got: #{inspect(other)}"
    end
  end

  @doc """
  Decodes a tag from a binary (for testing purposes).

  ## Examples

      tag = {make_ref(), self()}
      tensor = EXLA.FFI.TermTensor.tag_to_tensor(tag)
      bin = Nx.to_binary(tensor)
      ^tag = EXLA.FFI.TermTensor.binary_to_tag(bin)

  """
  @spec binary_to_tag(binary()) :: term()
  def binary_to_tag(bin) when byte_size(bin) == @tag_size do
    :erlang.binary_to_term(bin)
  end

  defp assert_size!(bin, expected, kind) do
    size = byte_size(bin)

    if size != expected do
      raise ArgumentError,
            "unexpected term_to_binary size for #{kind}: got #{size}, expected #{expected}"
    end
  end
end
