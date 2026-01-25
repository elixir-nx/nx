defmodule Nx.Floating do
  @moduledoc """
  Functions for loading and dumping floating-point formats
  not supported by the Erlang VM.
  """

  ## BF16

  @doc """
  Loads a BF16 (Brain Float 16) value from its binary representation.

  Returns a float, `:nan`, `:infinity`, or `:neg_infinity`.

  ## Examples

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(1.0))
      1.0

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(:infinity))
      :infinity

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(:neg_infinity))
      :neg_infinity

  """
  def load_bf16(<<0xFF80::16-native>>), do: :neg_infinity
  def load_bf16(<<0x7F80::16-native>>), do: :infinity

  if System.endianness() == :little do
    def load_bf16(<<1::1, _::7, _sign::1, 127::7>>), do: :nan

    def load_bf16(bf16) do
      <<x::float-little-32>> = <<0::16, bf16::binary>>
      x
    end
  else
    def load_bf16(<<_sign::1, 255::8, _::7>>), do: :nan

    def load_bf16(bf16) do
      <<x::float-big-32>> = <<bf16::binary, 0::16>>
      x
    end
  end

  @doc """
  Dumps an Elixir value to BF16 (Brain Float 16) binary representation.

  ## Examples

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(1.0))
      1.0

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(-2.5))
      -2.5

      iex> Nx.Floating.load_bf16(Nx.Floating.dump_bf16(:nan))
      :nan

  """
  if System.endianness() == :little do
    def dump_bf16(x) when is_number(x), do: binary_part(<<x::float-native-32>>, 2, 2)
  else
    def dump_bf16(x) when is_number(x), do: binary_part(<<x::float-native-32>>, 0, 2)
  end

  def dump_bf16(:infinity), do: unquote(Nx.Type.infinity_binary({:bf, 16}))
  def dump_bf16(:neg_infinity), do: unquote(Nx.Type.neg_infinity_binary({:bf, 16}))
  def dump_bf16(:nan), do: unquote(Nx.Type.nan_binary({:bf, 16}))

  ## F8 (E5M2)

  @doc """
  Loads an F8 (E5M2) value from its binary representation.

  Returns a float, `:nan`, `:infinity`, or `:neg_infinity`.

  ## Examples

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(1.0))
      1.0

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(:infinity))
      :infinity

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(:neg_infinity))
      :neg_infinity

  """
  def load_f8(<<0xFC::8-native>>), do: :neg_infinity
  def load_f8(<<0x7C::8-native>>), do: :infinity
  def load_f8(<<_sign::1, 31::5, mantissa::2>>) when mantissa != 0, do: :nan
  def load_f8(<<sign::1, 0::7>>), do: if(sign == 0, do: 0.0, else: -0.0)

  def load_f8(<<sign::1, exp::5, mantissa::2>>) do
    float = :math.pow(2, exp - 15) * (1 + mantissa / 4)

    case sign do
      0 -> float
      _ -> -float
    end
  end

  @doc """
  Dumps an Elixir value to F8 (E5M2) binary representation.

  ## Examples

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(1.0))
      1.0

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(-2.0))
      -2.0

      iex> Nx.Floating.load_f8(Nx.Floating.dump_f8(:nan))
      :nan

  """
  if System.endianness() == :little do
    def dump_f8(x) when is_number(x), do: binary_part(<<x::float-native-16>>, 1, 1)
  else
    def dump_f8(x) when is_number(x), do: binary_part(<<x::float-native-16>>, 0, 1)
  end

  def dump_f8(:infinity), do: unquote(Nx.Type.infinity_binary({:f, 8}))
  def dump_f8(:neg_infinity), do: unquote(Nx.Type.neg_infinity_binary({:f, 8}))
  def dump_f8(:nan), do: unquote(Nx.Type.nan_binary({:f, 8}))

  ## F8 E4M3FN

  @doc """
  Loads an F8 E4M3FN value from its binary representation.

  Returns a float or `:nan` (E4M3FN has no infinity).

  ## Examples

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(1.0))
      1.0

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(:nan))
      :nan

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(0.0))
      0.0

      iex> Nx.Floating.load_f8_e4m3fn(Nx.Floating.dump_f8_e4m3fn(448.0))
      448.0

  """
  # E4M3FN format: 1 sign bit, 4 exponent bits, 3 mantissa bits
  # Exponent bias: 7
  # Per OFP8 spec: E4M3FN has NO infinity (FN = "Finite, No infinities")
  # Only S.1111.111 is NaN; all other S.1111.xxx are finite values

  # Only mantissa = 111 (0x7) is NaN in E4M3FN
  def load_f8_e4m3fn(<<_sign::1, 15::4, 7::3>>), do: :nan

  def load_f8_e4m3fn(<<0::1, 0::4, mantissa::3>>) do
    # Denormalized positive
    :math.pow(2, -6) * (mantissa / 8.0)
  end

  def load_f8_e4m3fn(<<1::1, 0::4, mantissa::3>>) do
    # Denormalized negative
    -:math.pow(2, -6) * (mantissa / 8.0)
  end

  def load_f8_e4m3fn(<<sign::1, exp::4, mantissa::3>>) do
    # Normalized
    float = :math.pow(2, exp - 7) * (1 + mantissa / 8.0)

    case sign do
      0 -> float
      _ -> -float
    end
  end

  @doc """
  Dumps an Elixir value to F8 E4M3FN binary representation.

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
  # E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa
  # Max value: 448.0 (0x7E), Min value: -448.0 (0xFE)
  def dump_f8_e4m3fn(0), do: <<0b0000_0000>>
  def dump_f8_e4m3fn(+0.0), do: <<0b0000_0000>>
  def dump_f8_e4m3fn(-0.0), do: <<0b1000_0000>>

  def dump_f8_e4m3fn(x) when is_number(x) do
    # Clamp to E4M3FN range and convert
    # E4M3FN max is 448.0, min is -448.0
    clamped = max(-448.0, min(448.0, x * 1.0))

    # Extract sign
    {sign, abs_val} = if clamped < 0, do: {1, -clamped}, else: {0, clamped}

    # Calculate exponent and mantissa
    # E4M3FN: value = (1 + mantissa/8) * 2^(exp - 7) for normalized
    log2_val = :math.log2(abs_val)
    exp_unbiased = floor(log2_val)
    exp = exp_unbiased + 7

    cond do
      exp <= 0 ->
        # Denormalized: value = mantissa/8 * 2^(-6)
        mantissa = round(abs_val / :math.pow(2, -6) * 8)
        <<sign::1, 0::4, min(7, mantissa)::3>>

      exp > 15 ->
        # Overflow to max finite (not NaN, since E4M3FN saturates in our impl)
        <<sign::1, 15::4, 6::3>>

      true ->
        # Normalized: value = (1 + mantissa/8) * 2^(exp - 7)
        significand = abs_val / :math.pow(2, exp_unbiased)
        mantissa = round((significand - 1.0) * 8)
        mantissa = max(0, min(7, mantissa))

        # exp=15 with mantissa=7 would be NaN, cap at 6
        if exp == 15 and mantissa >= 7 do
          <<sign::1, 15::4, 6::3>>
        else
          <<sign::1, exp::4, mantissa::3>>
        end
    end
  end

  # Saturate infinity to max/min finite values (not NaN)
  # This preserves sign and matches overflow saturation behavior
  def dump_f8_e4m3fn(:infinity), do: unquote(Nx.Type.infinity_binary({:f8_e4m3fn, 8}))
  def dump_f8_e4m3fn(:neg_infinity), do: unquote(Nx.Type.neg_infinity_binary({:f8_e4m3fn, 8}))
  def dump_f8_e4m3fn(:nan), do: unquote(Nx.Type.nan_binary({:f8_e4m3fn, 8}))
end
