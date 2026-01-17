defmodule Nx.Floating do
  @moduledoc """
  Functions for loading and dumping special floating-point formats.

  This module provides conversion functions between Elixir values
  and binary representations of floating-point formats that are not
  natively supported by the BEAM.
  """

  @doc """
  Loads an f8 E4M3FN value from its binary representation.

  Returns a float or `:nan` (E4M3FN has no infinity).

  ## Examples

      iex> Nx.Floating.load_f8_e4m3fn(<<0x38>>)
      1.0

      iex> Nx.Floating.load_f8_e4m3fn(<<0x7F>>)
      :nan

      iex> Nx.Floating.load_f8_e4m3fn(<<0x00>>)
      0.0

      iex> Nx.Floating.load_f8_e4m3fn(<<0x7E>>)
      448.0

  """
  def load_f8_e4m3fn(<<byte::8>>) do
    Nx.Shared.read_f8_e4m3fn(<<byte::8>>)
  end

  @doc """
  Dumps an Elixir value to f8 E4M3FN binary representation.

  Finite values are clamped to the E4M3FN range [-448.0, 448.0].
  Since E4M3FN has no infinity, `:infinity` and `:neg_infinity` saturate
  to max/min finite values. Only `:nan` maps to NaN.

  ## Examples

      iex> Nx.Floating.dump_f8_e4m3fn(1.0)
      <<0x38>>

      iex> Nx.Floating.dump_f8_e4m3fn(0.0)
      <<0x00>>

      iex> Nx.Floating.dump_f8_e4m3fn(448.0)
      <<0x7E>>

      iex> Nx.Floating.dump_f8_e4m3fn(-448.0)
      <<0xFE>>

      iex> Nx.Floating.dump_f8_e4m3fn(:infinity)
      <<0x7E>>

      iex> Nx.Floating.dump_f8_e4m3fn(:neg_infinity)
      <<0xFE>>

      iex> Nx.Floating.dump_f8_e4m3fn(:nan)
      <<0x7F>>

  """
  def dump_f8_e4m3fn(value) when is_number(value) do
    Nx.Shared.write_finite_f8_e4m3fn(value)
  end

  # Saturate infinity to max/min finite values (not NaN)
  # This preserves sign and matches overflow saturation behavior
  def dump_f8_e4m3fn(:infinity), do: <<0x7E>>
  def dump_f8_e4m3fn(:neg_infinity), do: <<0xFE>>
  def dump_f8_e4m3fn(:nan), do: <<0x7F>>
end
