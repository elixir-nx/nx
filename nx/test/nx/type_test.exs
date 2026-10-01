defmodule Nx.TypeTest do
  use ExUnit.Case, async: true

  doctest Nx.Type

  test "normalize! raises on an unknown type" do
    assert_raise ArgumentError, ~r/invalid numerical type: \{:k, 8\}/, fn ->
      apply(Nx.Type, :normalize!, [{:k, 8}])
    end
  end

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

    test "is not recognized as complex type" do
      assert Nx.Type.complex?({:f8_e4m3fn, 8}) == false
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

    test "infinity_binary saturates to max finite (0x7E) since E4M3FN has no infinity" do
      # E4M3FN has no infinity per OFP8 spec (FN = "Finite, No infinities")
      # Saturate to max finite value for consistency with overflow behavior
      assert Nx.Type.infinity_binary({:f8_e4m3fn, 8}) == <<0x7E::8-native>>
    end

    test "neg_infinity_binary saturates to min finite (0xFE) since E4M3FN has no infinity" do
      # Saturate to min finite value for consistency with overflow behavior
      assert Nx.Type.neg_infinity_binary({:f8_e4m3fn, 8}) == <<0xFE::8-native>>
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

  describe "merge/2" do
    test "merges a quantized type into a wider type of its base type in either order" do
      for type <- [f: 16, f: 32, f: 64] do
        assert Nx.Type.merge({:f8_e4m3fn, 8}, type) == type
        assert Nx.Type.merge(type, {:f8_e4m3fn, 8}) == type
      end
    end

    test "raises on different types with the same base type and size in either order" do
      assert_raise ArgumentError,
                   "cannot merge {:f8_e4m3fn, 8} and {:f, 8}, convert one of them with Nx.as_type/2 first",
                   fn -> Nx.Type.merge({:f8_e4m3fn, 8}, {:f, 8}) end

      assert_raise ArgumentError,
                   "cannot merge {:f, 8} and {:f8_e4m3fn, 8}, convert one of them with Nx.as_type/2 first",
                   fn -> Nx.Type.merge({:f, 8}, {:f8_e4m3fn, 8}) end
    end

    test "gives a quantized type the precedence of its base type in either order" do
      for {type, expected} <- [
            {{:u, 8}, {:f8_e4m3fn, 8}},
            {{:s, 64}, {:f8_e4m3fn, 8}},
            {{:bf, 16}, {:f8_e4m3fn, 8}},
            {{:c, 64}, {:c, 64}},
            {{:c, 128}, {:c, 128}}
          ] do
        assert Nx.Type.merge({:f8_e4m3fn, 8}, type) == expected
        assert Nx.Type.merge(type, {:f8_e4m3fn, 8}) == expected
      end
    end

    test "returns a quantized type unchanged when merged with itself" do
      assert Nx.Type.merge({:f8_e4m3fn, 8}, {:f8_e4m3fn, 8}) == {:f8_e4m3fn, 8}
    end
  end
end
