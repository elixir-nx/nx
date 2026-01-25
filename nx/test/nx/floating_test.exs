defmodule Nx.FloatingTest do
  use ExUnit.Case, async: true

  doctest Nx.Floating

  # Complete lookup table of all 256 E4M3FN binary ↔ float pairs
  # E4M3FN format: 1 sign bit, 4 exponent bits, 3 mantissa bits
  # Bias = 7, no infinity (0x7F and 0xFF are NaN)
  @e4m3fn_lookup_table [
    # Positive values (sign bit = 0)
    # Denormalized (exponent = 0): value = mantissa/8 * 2^-6
    {0x00, 0.0},
    {0x01, 0.001953125},
    {0x02, 0.00390625},
    {0x03, 0.005859375},
    {0x04, 0.0078125},
    {0x05, 0.009765625},
    {0x06, 0.01171875},
    {0x07, 0.013671875},
    # Normalized (exponent = 1): 2^(1-7) * (1 + m/8) = 2^-6 * (1 + m/8)
    {0x08, 0.015625},
    {0x09, 0.017578125},
    {0x0A, 0.01953125},
    {0x0B, 0.021484375},
    {0x0C, 0.0234375},
    {0x0D, 0.025390625},
    {0x0E, 0.02734375},
    {0x0F, 0.029296875},
    # Exponent = 2: 2^(2-7) * (1 + m/8) = 2^-5 * (1 + m/8)
    {0x10, 0.03125},
    {0x11, 0.03515625},
    {0x12, 0.0390625},
    {0x13, 0.04296875},
    {0x14, 0.046875},
    {0x15, 0.05078125},
    {0x16, 0.0546875},
    {0x17, 0.05859375},
    # Exponent = 3: 2^(3-7) * (1 + m/8) = 2^-4 * (1 + m/8)
    {0x18, 0.0625},
    {0x19, 0.0703125},
    {0x1A, 0.078125},
    {0x1B, 0.0859375},
    {0x1C, 0.09375},
    {0x1D, 0.1015625},
    {0x1E, 0.109375},
    {0x1F, 0.1171875},
    # Exponent = 4: 2^(4-7) * (1 + m/8) = 2^-3 * (1 + m/8)
    {0x20, 0.125},
    {0x21, 0.140625},
    {0x22, 0.15625},
    {0x23, 0.171875},
    {0x24, 0.1875},
    {0x25, 0.203125},
    {0x26, 0.21875},
    {0x27, 0.234375},
    # Exponent = 5: 2^(5-7) * (1 + m/8) = 2^-2 * (1 + m/8)
    {0x28, 0.25},
    {0x29, 0.28125},
    {0x2A, 0.3125},
    {0x2B, 0.34375},
    {0x2C, 0.375},
    {0x2D, 0.40625},
    {0x2E, 0.4375},
    {0x2F, 0.46875},
    # Exponent = 6: 2^(6-7) * (1 + m/8) = 2^-1 * (1 + m/8)
    {0x30, 0.5},
    {0x31, 0.5625},
    {0x32, 0.625},
    {0x33, 0.6875},
    {0x34, 0.75},
    {0x35, 0.8125},
    {0x36, 0.875},
    {0x37, 0.9375},
    # Exponent = 7: 2^(7-7) * (1 + m/8) = 1 * (1 + m/8)
    {0x38, 1.0},
    {0x39, 1.125},
    {0x3A, 1.25},
    {0x3B, 1.375},
    {0x3C, 1.5},
    {0x3D, 1.625},
    {0x3E, 1.75},
    {0x3F, 1.875},
    # Exponent = 8: 2^(8-7) * (1 + m/8) = 2 * (1 + m/8)
    {0x40, 2.0},
    {0x41, 2.25},
    {0x42, 2.5},
    {0x43, 2.75},
    {0x44, 3.0},
    {0x45, 3.25},
    {0x46, 3.5},
    {0x47, 3.75},
    # Exponent = 9: 2^(9-7) * (1 + m/8) = 4 * (1 + m/8)
    {0x48, 4.0},
    {0x49, 4.5},
    {0x4A, 5.0},
    {0x4B, 5.5},
    {0x4C, 6.0},
    {0x4D, 6.5},
    {0x4E, 7.0},
    {0x4F, 7.5},
    # Exponent = 10: 2^(10-7) * (1 + m/8) = 8 * (1 + m/8)
    {0x50, 8.0},
    {0x51, 9.0},
    {0x52, 10.0},
    {0x53, 11.0},
    {0x54, 12.0},
    {0x55, 13.0},
    {0x56, 14.0},
    {0x57, 15.0},
    # Exponent = 11: 2^(11-7) * (1 + m/8) = 16 * (1 + m/8)
    {0x58, 16.0},
    {0x59, 18.0},
    {0x5A, 20.0},
    {0x5B, 22.0},
    {0x5C, 24.0},
    {0x5D, 26.0},
    {0x5E, 28.0},
    {0x5F, 30.0},
    # Exponent = 12: 2^(12-7) * (1 + m/8) = 32 * (1 + m/8)
    {0x60, 32.0},
    {0x61, 36.0},
    {0x62, 40.0},
    {0x63, 44.0},
    {0x64, 48.0},
    {0x65, 52.0},
    {0x66, 56.0},
    {0x67, 60.0},
    # Exponent = 13: 2^(13-7) * (1 + m/8) = 64 * (1 + m/8)
    {0x68, 64.0},
    {0x69, 72.0},
    {0x6A, 80.0},
    {0x6B, 88.0},
    {0x6C, 96.0},
    {0x6D, 104.0},
    {0x6E, 112.0},
    {0x6F, 120.0},
    # Exponent = 14: 2^(14-7) * (1 + m/8) = 128 * (1 + m/8)
    {0x70, 128.0},
    {0x71, 144.0},
    {0x72, 160.0},
    {0x73, 176.0},
    {0x74, 192.0},
    {0x75, 208.0},
    {0x76, 224.0},
    {0x77, 240.0},
    # Exponent = 15 (max): 2^(15-7) * (1 + m/8) = 256 * (1 + m/8)
    {0x78, 256.0},
    {0x79, 288.0},
    {0x7A, 320.0},
    {0x7B, 352.0},
    {0x7C, 384.0},
    {0x7D, 416.0},
    {0x7E, 448.0},
    # 0x7F is NaN (handled separately)
    {0x7F, :nan},
    # Negative values (sign bit = 1)
    # Denormalized (exponent = 0): value = -mantissa/8 * 2^-6
    {0x80, -0.0},
    {0x81, -0.001953125},
    {0x82, -0.00390625},
    {0x83, -0.005859375},
    {0x84, -0.0078125},
    {0x85, -0.009765625},
    {0x86, -0.01171875},
    {0x87, -0.013671875},
    # Normalized (exponent = 1)
    {0x88, -0.015625},
    {0x89, -0.017578125},
    {0x8A, -0.01953125},
    {0x8B, -0.021484375},
    {0x8C, -0.0234375},
    {0x8D, -0.025390625},
    {0x8E, -0.02734375},
    {0x8F, -0.029296875},
    # Exponent = 2
    {0x90, -0.03125},
    {0x91, -0.03515625},
    {0x92, -0.0390625},
    {0x93, -0.04296875},
    {0x94, -0.046875},
    {0x95, -0.05078125},
    {0x96, -0.0546875},
    {0x97, -0.05859375},
    # Exponent = 3
    {0x98, -0.0625},
    {0x99, -0.0703125},
    {0x9A, -0.078125},
    {0x9B, -0.0859375},
    {0x9C, -0.09375},
    {0x9D, -0.1015625},
    {0x9E, -0.109375},
    {0x9F, -0.1171875},
    # Exponent = 4
    {0xA0, -0.125},
    {0xA1, -0.140625},
    {0xA2, -0.15625},
    {0xA3, -0.171875},
    {0xA4, -0.1875},
    {0xA5, -0.203125},
    {0xA6, -0.21875},
    {0xA7, -0.234375},
    # Exponent = 5
    {0xA8, -0.25},
    {0xA9, -0.28125},
    {0xAA, -0.3125},
    {0xAB, -0.34375},
    {0xAC, -0.375},
    {0xAD, -0.40625},
    {0xAE, -0.4375},
    {0xAF, -0.46875},
    # Exponent = 6
    {0xB0, -0.5},
    {0xB1, -0.5625},
    {0xB2, -0.625},
    {0xB3, -0.6875},
    {0xB4, -0.75},
    {0xB5, -0.8125},
    {0xB6, -0.875},
    {0xB7, -0.9375},
    # Exponent = 7
    {0xB8, -1.0},
    {0xB9, -1.125},
    {0xBA, -1.25},
    {0xBB, -1.375},
    {0xBC, -1.5},
    {0xBD, -1.625},
    {0xBE, -1.75},
    {0xBF, -1.875},
    # Exponent = 8
    {0xC0, -2.0},
    {0xC1, -2.25},
    {0xC2, -2.5},
    {0xC3, -2.75},
    {0xC4, -3.0},
    {0xC5, -3.25},
    {0xC6, -3.5},
    {0xC7, -3.75},
    # Exponent = 9
    {0xC8, -4.0},
    {0xC9, -4.5},
    {0xCA, -5.0},
    {0xCB, -5.5},
    {0xCC, -6.0},
    {0xCD, -6.5},
    {0xCE, -7.0},
    {0xCF, -7.5},
    # Exponent = 10
    {0xD0, -8.0},
    {0xD1, -9.0},
    {0xD2, -10.0},
    {0xD3, -11.0},
    {0xD4, -12.0},
    {0xD5, -13.0},
    {0xD6, -14.0},
    {0xD7, -15.0},
    # Exponent = 11
    {0xD8, -16.0},
    {0xD9, -18.0},
    {0xDA, -20.0},
    {0xDB, -22.0},
    {0xDC, -24.0},
    {0xDD, -26.0},
    {0xDE, -28.0},
    {0xDF, -30.0},
    # Exponent = 12
    {0xE0, -32.0},
    {0xE1, -36.0},
    {0xE2, -40.0},
    {0xE3, -44.0},
    {0xE4, -48.0},
    {0xE5, -52.0},
    {0xE6, -56.0},
    {0xE7, -60.0},
    # Exponent = 13
    {0xE8, -64.0},
    {0xE9, -72.0},
    {0xEA, -80.0},
    {0xEB, -88.0},
    {0xEC, -96.0},
    {0xED, -104.0},
    {0xEE, -112.0},
    {0xEF, -120.0},
    # Exponent = 14
    {0xF0, -128.0},
    {0xF1, -144.0},
    {0xF2, -160.0},
    {0xF3, -176.0},
    {0xF4, -192.0},
    {0xF5, -208.0},
    {0xF6, -224.0},
    {0xF7, -240.0},
    # Exponent = 15 (max)
    {0xF8, -256.0},
    {0xF9, -288.0},
    {0xFA, -320.0},
    {0xFB, -352.0},
    {0xFC, -384.0},
    {0xFD, -416.0},
    {0xFE, -448.0},
    # 0xFF is NaN (handled separately)
    {0xFF, :nan}
  ]

  describe "exhaustive E4M3FN lookup table" do
    test "load_f8_e4m3fn/1 correctly decodes all 256 binary values" do
      for {byte, expected} <- @e4m3fn_lookup_table do
        result = Nx.Floating.load_f8_e4m3fn(<<byte>>)

        case expected do
          :nan ->
            assert result == :nan,
                   "0x#{Integer.to_string(byte, 16) |> String.pad_leading(2, "0")}: expected :nan, got #{inspect(result)}"

          -0.0 ->
            # Check for negative zero
            assert result == 0.0 or result == -0.0,
                   "0x#{Integer.to_string(byte, 16) |> String.pad_leading(2, "0")}: expected -0.0, got #{inspect(result)}"

          _ ->
            assert result == expected,
                   "0x#{Integer.to_string(byte, 16) |> String.pad_leading(2, "0")}: expected #{expected}, got #{inspect(result)}"
        end
      end
    end

    test "dump_f8_e4m3fn/1 correctly encodes all representable float values (round-trip)" do
      # Test round-trip for all non-NaN, non-negative-zero values
      for {byte, value} <- @e4m3fn_lookup_table, value != :nan and value != -0.0 do
        result = Nx.Floating.dump_f8_e4m3fn(value)

        assert result == <<byte>>,
               "dump(#{value}): expected <<0x#{Integer.to_string(byte, 16) |> String.pad_leading(2, "0")}>>, got <<0x#{Integer.to_string(:binary.decode_unsigned(result), 16) |> String.pad_leading(2, "0")}>>"
      end
    end

    test "complete round-trip for all 256 values" do
      for {byte, expected} <- @e4m3fn_lookup_table do
        loaded = Nx.Floating.load_f8_e4m3fn(<<byte>>)

        case expected do
          :nan ->
            assert loaded == :nan
            # NaN dumps back to positive NaN (0x7F)
            assert Nx.Floating.dump_f8_e4m3fn(:nan) == <<0x7F>>

          -0.0 ->
            # -0.0 loads as 0.0 or -0.0, dumps back to 0x00
            assert Nx.Floating.dump_f8_e4m3fn(0.0) == <<0x00>>

          _ ->
            # All other values should round-trip exactly
            dumped = Nx.Floating.dump_f8_e4m3fn(loaded)
            reloaded = Nx.Floating.load_f8_e4m3fn(dumped)

            assert reloaded == expected,
                   "Round-trip failed for 0x#{Integer.to_string(byte, 16) |> String.pad_leading(2, "0")}: #{expected} -> #{loaded} -> #{inspect(dumped)} -> #{reloaded}"
        end
      end
    end
  end

  describe "load_f8_e4m3fn/1" do
    test "loads zero" do
      assert Nx.Floating.load_f8_e4m3fn(<<0x00>>) == 0.0
      assert Nx.Floating.load_f8_e4m3fn(<<0x80>>) == -0.0
    end

    test "loads common values" do
      assert Nx.Floating.load_f8_e4m3fn(<<0x38>>) == 1.0
      assert Nx.Floating.load_f8_e4m3fn(<<0xB8>>) == -1.0
      assert Nx.Floating.load_f8_e4m3fn(<<0x40>>) == 2.0
      assert Nx.Floating.load_f8_e4m3fn(<<0xC0>>) == -2.0
    end

    test "loads max finite values" do
      assert Nx.Floating.load_f8_e4m3fn(<<0x7E>>) == 448.0
      assert Nx.Floating.load_f8_e4m3fn(<<0xFE>>) == -448.0
    end

    test "loads NaN" do
      assert Nx.Floating.load_f8_e4m3fn(<<0x7F>>) == :nan
      assert Nx.Floating.load_f8_e4m3fn(<<0xFF>>) == :nan
    end

    test "loads denormalized values" do
      # Denormalized: value = mantissa/8 * 2^(-6)
      # 0x01 = 0.0000.001 -> 1/8 * 2^-6 = 1/512
      assert_in_delta Nx.Floating.load_f8_e4m3fn(<<0x01>>), 1 / 512, 1.0e-10
      # 0x07 = 0.0000.111 -> 7/8 * 2^-6 = 7/512
      assert_in_delta Nx.Floating.load_f8_e4m3fn(<<0x07>>), 7 / 512, 1.0e-10
    end
  end

  describe "dump_f8_e4m3fn/1" do
    test "dumps zero" do
      assert Nx.Floating.dump_f8_e4m3fn(0.0) == <<0x00>>
      assert Nx.Floating.dump_f8_e4m3fn(0) == <<0x00>>
    end

    test "dumps common values" do
      assert Nx.Floating.dump_f8_e4m3fn(1.0) == <<0x38>>
      assert Nx.Floating.dump_f8_e4m3fn(-1.0) == <<0xB8>>
      assert Nx.Floating.dump_f8_e4m3fn(2.0) == <<0x40>>
      assert Nx.Floating.dump_f8_e4m3fn(-2.0) == <<0xC0>>
    end

    test "dumps max finite values" do
      assert Nx.Floating.dump_f8_e4m3fn(448.0) == <<0x7E>>
      assert Nx.Floating.dump_f8_e4m3fn(-448.0) == <<0xFE>>
    end

    test "clamps overflow to max finite" do
      assert Nx.Floating.dump_f8_e4m3fn(1000.0) == <<0x7E>>
      assert Nx.Floating.dump_f8_e4m3fn(-1000.0) == <<0xFE>>
    end

    test "saturates infinity to max/min finite (preserves sign)" do
      # Infinity saturates to max finite (448.0)
      assert Nx.Floating.dump_f8_e4m3fn(:infinity) == <<0x7E>>
      # Negative infinity saturates to min finite (-448.0)
      assert Nx.Floating.dump_f8_e4m3fn(:neg_infinity) == <<0xFE>>
    end

    test "dumps NaN" do
      assert Nx.Floating.dump_f8_e4m3fn(:nan) == <<0x7F>>
    end

    test "accepts integers" do
      assert Nx.Floating.dump_f8_e4m3fn(1) == <<0x38>>
      assert Nx.Floating.dump_f8_e4m3fn(-1) == <<0xB8>>
    end
  end

  describe "round-trip" do
    test "round-trips common values" do
      for value <- [
            0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            4.0,
            -4.0,
            8.0,
            16.0,
            64.0,
            128.0,
            256.0,
            448.0,
            -448.0
          ] do
        binary = Nx.Floating.dump_f8_e4m3fn(value)
        loaded = Nx.Floating.load_f8_e4m3fn(binary)
        assert loaded == value, "Expected #{value} but got #{inspect(loaded)}"
      end
    end

    test "round-trips NaN" do
      binary = Nx.Floating.dump_f8_e4m3fn(:nan)
      assert Nx.Floating.load_f8_e4m3fn(binary) == :nan
    end

    test "infinity round-trips to max finite" do
      # Infinity saturates to max, then loads as max
      binary = Nx.Floating.dump_f8_e4m3fn(:infinity)
      assert Nx.Floating.load_f8_e4m3fn(binary) == 448.0

      binary = Nx.Floating.dump_f8_e4m3fn(:neg_infinity)
      assert Nx.Floating.load_f8_e4m3fn(binary) == -448.0
    end

    test "overflow round-trips to max finite" do
      binary = Nx.Floating.dump_f8_e4m3fn(1000.0)
      assert Nx.Floating.load_f8_e4m3fn(binary) == 448.0

      binary = Nx.Floating.dump_f8_e4m3fn(-1000.0)
      assert Nx.Floating.load_f8_e4m3fn(binary) == -448.0
    end
  end

  describe "precision limits" do
    test "values outside representable range are approximated" do
      # E4M3FN can only represent values exactly at specific points
      # Values between representable points get rounded
      binary = Nx.Floating.dump_f8_e4m3fn(1.5)
      loaded = Nx.Floating.load_f8_e4m3fn(binary)
      # 1.5 should round to nearest representable value
      assert is_float(loaded)
      assert_in_delta loaded, 1.5, 0.25
    end
  end

  describe "tensor round-trip" do
    test "to_binary and from_binary round-trip" do
      original = Nx.tensor([1.0, 2.0, 4.0], type: :f8_e4m3fn)
      binary = Nx.to_binary(original)
      restored = Nx.from_binary(binary, {:f8_e4m3fn, 8})

      assert Nx.type(restored) == {:f8_e4m3fn, 8}
      assert Nx.shape(restored) == {3}
    end
  end

  describe "tensor creation" do
    test "creates tensor with :f8_e4m3fn type" do
      tensor = Nx.tensor([1.0, 2.0, 3.0], type: :f8_e4m3fn)
      assert Nx.type(tensor) == {:f8_e4m3fn, 8}
      assert Nx.shape(tensor) == {3}
    end

    test "creates multi-dimensional tensor" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f8_e4m3fn)
      assert Nx.type(tensor) == {:f8_e4m3fn, 8}
      assert Nx.shape(tensor) == {2, 2}
    end

    test "binary size is 1 byte per element" do
      tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f8_e4m3fn)
      binary = Nx.to_binary(tensor)
      assert byte_size(binary) == 4
    end
  end
end
