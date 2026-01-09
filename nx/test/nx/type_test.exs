defmodule Nx.TypeTest do
  use ExUnit.Case, async: true

  doctest Nx.Type

  describe "fp8 E4M3FN type" do
    test "normalizes :f8_e4m3fn atom to tuple" do
      assert Nx.Type.normalize!(:f8_e4m3fn) == {:f8_e4m3fn, 8}
    end

    test "normalizes {:f8_e4m3fn, 8} tuple" do
      assert Nx.Type.normalize!({:f8_e4m3fn, 8}) == {:f8_e4m3fn, 8}
    end

    test "is recognized as float type" do
      assert Nx.Type.float?({:f8_e4m3fn, 8}) == true
    end

    test "is not recognized as integer type" do
      assert Nx.Type.integer?({:f8_e4m3fn, 8}) == false
    end

    test "to_string returns correct representation" do
      assert Nx.Type.to_string({:f8_e4m3fn, 8}) == "f8_e4m3fn"
    end

    test "to_floating preserves type" do
      assert Nx.Type.to_floating({:f8_e4m3fn, 8}) == {:f8_e4m3fn, 8}
    end

    test "to_real preserves type" do
      assert Nx.Type.to_real({:f8_e4m3fn, 8}) == {:f8_e4m3fn, 8}
    end
  end

  describe "fp8 E4M3FN special values (per OFP8 spec)" do
    test "max_finite_binary returns 0x7E (448.0)" do
      assert Nx.Type.max_finite_binary({:f8_e4m3fn, 8}) == <<0x7E::8-native>>
    end

    test "min_finite_binary returns 0xFE (-448.0)" do
      assert Nx.Type.min_finite_binary({:f8_e4m3fn, 8}) == <<0xFE::8-native>>
    end

    test "nan_binary returns 0x7F" do
      assert Nx.Type.nan_binary({:f8_e4m3fn, 8}) == <<0x7F::8-native>>
    end

    test "infinity_binary returns NaN (0x7F) since E4M3FN has no infinity" do
      # E4M3FN has no infinity per OFP8 spec (FN = "Finite, No infinities")
      assert Nx.Type.infinity_binary({:f8_e4m3fn, 8}) == <<0x7F::8-native>>
    end

    test "neg_infinity_binary returns NaN (0x7F) since E4M3FN has no infinity" do
      assert Nx.Type.neg_infinity_binary({:f8_e4m3fn, 8}) == <<0x7F::8-native>>
    end

    test "max_binary returns finite max (not infinity)" do
      # Since E4M3FN has no infinity, max_binary returns max_finite_binary
      assert Nx.Type.max_binary({:f8_e4m3fn, 8}) == <<0x7E::8-native>>
    end

    test "min_binary returns finite min (not neg_infinity)" do
      # Since E4M3FN has no infinity, min_binary returns min_finite_binary
      assert Nx.Type.min_binary({:f8_e4m3fn, 8}) == <<0xFE::8-native>>
    end

    test "smallest_positive_normal_binary returns 0x08" do
      assert Nx.Type.smallest_positive_normal_binary({:f8_e4m3fn, 8}) == <<0x08::8-native>>
    end
  end

  describe "fp8 E4M3FN tensor creation" do
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

  describe "fp8 E4M3FN binary round-trip" do
    test "to_binary and from_binary round-trip" do
      original = Nx.tensor([1.0, 2.0, 4.0], type: :f8_e4m3fn)
      binary = Nx.to_binary(original)
      restored = Nx.from_binary(binary, {:f8_e4m3fn, 8})

      assert Nx.type(restored) == {:f8_e4m3fn, 8}
      assert Nx.shape(restored) == {3}
    end
  end

  describe "existing {:f, 8} type (E5M2) unchanged" do
    test "validates {:f, 8} tuple" do
      assert Nx.Type.normalize!({:f, 8}) == {:f, 8}
    end

    test "validates :f8 atom" do
      assert Nx.Type.normalize!(:f8) == {:f, 8}
    end

    test "E5M2 has infinity (0x7C)" do
      assert Nx.Type.infinity_binary({:f, 8}) == <<0x7C::8-native>>
    end

    test "E5M2 has neg_infinity (0xFC)" do
      assert Nx.Type.neg_infinity_binary({:f, 8}) == <<0xFC::8-native>>
    end
  end
end
