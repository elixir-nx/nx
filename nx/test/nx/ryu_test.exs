defmodule Nx.RyuTest do
  use ExUnit.Case, async: true

  import Nx.Ryu, only: [bits_to_decimal: 3]

  defp f16_to_decimal(bits), do: bits_to_decimal(bits, 10, 5)
  defp f32_to_decimal(bits), do: bits_to_decimal(bits, 23, 8)

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
end
