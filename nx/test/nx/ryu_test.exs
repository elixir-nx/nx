defmodule Nx.RyuTest do
  use ExUnit.Case, async: true

  import Nx.Ryu, only: [bits_to_decimal: 3, bits_to_decimal: 4]

  defp f16_to_decimal(bits), do: bits_to_decimal(bits, 10, 5)
  defp f32_to_decimal(bits), do: bits_to_decimal(bits, 23, 8)
  defp f8_e4m3fn_to_decimal(bits), do: bits_to_decimal(bits, 3, 4, :fn)

  test "f16: matches C Ryu output for all 65536 values" do
    mismatches =
      File.read!("test/fixtures/f16_table.txt")
      |> String.split("\n", trim: true)
      |> Enum.filter(fn line ->
        <<hex::binary-size(4), ?:, value::binary>> = line
        bits = String.to_integer(hex, 16)

        # Normalize C output: -NaN -> NaN, Infinity -> Inf, E -> e
        c_output =
          value
          |> String.replace("-NaN", "NaN")
          |> String.replace("Infinity", "Inf")
          |> String.replace("E", "e")

        elixir_output = f16_to_decimal(bits)

        if c_output != elixir_output do
          {bits, c_output, elixir_output}
        end
      end)

    if mismatches != [] do
      IO.puts("\n\nFirst 10 mismatches:")
      IO.puts("Bits  | C Output       | Elixir Output  ")
      IO.puts("------|----------------|----------------")

      mismatches
      |> Enum.take(10)
      |> Enum.each(fn {bits, c, elixir} ->
        hex = bits |> Integer.to_string(16) |> String.upcase() |> String.pad_leading(4, "0")
        IO.puts("#{hex}  | #{String.pad_trailing(c, 14)} | #{elixir}")
      end)

      flunk("#{length(mismatches)} out of 65536 values don't match (see above)")
    end
  end

  test "f32" do
    # Zero
    assert f32_to_decimal(0x00000000) == "0.0"
    assert f32_to_decimal(0x80000000) == "-0.0"

    # Simple values
    assert f32_to_decimal(0x3F800000) == "1.0"
    assert f32_to_decimal(0x40000000) == "2.0"
    assert f32_to_decimal(0x3DCCCCCD) == "0.1"

    # Infinity
    assert f32_to_decimal(0x7F800000) == "Inf"
    assert f32_to_decimal(0xFF800000) == "-Inf"

    # NaN
    assert f32_to_decimal(0x7F800001) == "NaN"
    assert f32_to_decimal(0xFF800001) == "NaN"
  end

  test "f8_e4m3fn with :fn modifier" do
    # Zero
    assert f8_e4m3fn_to_decimal(0x00) == "0.0"
    assert f8_e4m3fn_to_decimal(0x80) == "-0.0"

    # NaN only when all mantissa bits are 1 (mantissa=7)
    # 0x7F = 0_1111_111 (sign=0, exp=15, mantissa=7) = positive NaN
    assert f8_e4m3fn_to_decimal(0x7F) == "NaN"
    # 0xFF = 1_1111_111 (sign=1, exp=15, mantissa=7) = negative NaN
    assert f8_e4m3fn_to_decimal(0xFF) == "NaN"

    # Values that would be Inf in standard IEEE 754 are normal numbers in FN
    # 0x78 = 0_1111_000 (sign=0, exp=15, mantissa=0) = would be +Inf, but is 448.0 in FN
    assert f8_e4m3fn_to_decimal(0x78) != "Inf"
    assert f8_e4m3fn_to_decimal(0x78) =~ ~r/^\d/
    # 0xF8 = 1_1111_000 (sign=1, exp=15, mantissa=0) = would be -Inf, but is -448.0 in FN
    assert f8_e4m3fn_to_decimal(0xF8) != "-Inf"
    assert f8_e4m3fn_to_decimal(0xF8) =~ ~r/^-\d/

    # Values that would be NaN in standard IEEE 754 (exp=max, mantissa!=0 but not all 1s)
    # are normal numbers in FN
    # 0x79 = 0_1111_001 (sign=0, exp=15, mantissa=1) = would be NaN, but is a number in FN
    assert f8_e4m3fn_to_decimal(0x79) != "NaN"
    assert f8_e4m3fn_to_decimal(0x79) =~ ~r/^\d/
    # 0x7E = 0_1111_110 (sign=0, exp=15, mantissa=6) = would be NaN, but is a number in FN
    assert f8_e4m3fn_to_decimal(0x7E) != "NaN"
    assert f8_e4m3fn_to_decimal(0x7E) =~ ~r/^\d/
  end
end
