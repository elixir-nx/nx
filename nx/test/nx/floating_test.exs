defmodule Nx.FloatingTest do
  use ExUnit.Case, async: true

  doctest Nx.Floating

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
end
