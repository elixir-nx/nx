defmodule Nx.BinaryBackend.BitsTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.Bits

  doctest Bits

  @float_types [{:f, 64}, {:f, 32}, {:bf, 16}]
  @int_types [{:s, 64}, {:s, 32}, {:u, 64}, {:u, 32}, {:u, 16}]

  defguardp is_float_type(t) when t in @float_types

  defguardp is_int_type(t) when t in @int_types

  defp epsilon({:bf, 16}), do: 1.0
  defp epsilon({:f, 32}), do: 0.001
  defp epsilon({:f, 64}), do: 0.0000001

  defp rand_float(_t \\ nil) do
    :rand.uniform() * 200.0 - 100.0
  end

  defp rand_int(t) do
    low = min_val(t)
    high = max_val(t)
    range = high - low
    :rand.uniform(range) - low
  end

  defp rand_num(t) when is_int_type(t) do
    rand_int(t)
  end

  defp rand_num(t) when is_float_type(t) do
    rand_float(t)
  end

  defp rand_num_encodable(t) do
    t
    |> rand_num()
    |> Bits.from_number(t)
    |> Bits.to_number(t)
  end

  defp min_val({:u, _}), do: 0
  defp min_val(_), do: -10_000

  defp max_val(_), do: 10_000

  defp zero(t), do: 0 * one(t)

  defp one(t) when is_float_type(t), do: 1.0
  defp one(t) when is_int_type(t), do: 1

  describe "from_number/2 and to_number/2" do
    test "can encode and decode all float types" do
      for t <- @float_types do
        n1 = rand_float()
        b = Bits.from_number(n1, t)
        n2 = Bits.to_number(b, t)
        eps = epsilon(t)

        assert_in_delta(
          n1,
          n2,
          eps,
          "type #{inspect(t)} was significantly different (eps: #{eps}) - before: #{n1} - after: #{
            n2
          }"
        )
      end
    end

    test "can encode and decode all integer types" do
      for t <- @int_types do
        n1 = rand_int(t)
        b = Bits.from_number(n1, t)
        n2 = Bits.to_number(b, t)

        assert n1 == n2,
               "int type #{inspect(t)} encode/decode failed - before: #{n1} - after: #{n2}"
      end
    end
  end

  describe "number_at/3" do
    test "works for all types" do
      for t <- @int_types ++ @float_types do
        n = rand_num_encodable(t)
        zero = zero(t)
        one = one(t)
        assert zero == 0
        assert one == 1

        bin =
          [n, zero, one]
          |> Enum.map(fn x -> Bits.from_number(x, t) end)
          |> IO.iodata_to_binary()

        assert Bits.number_at(bin, t, 0) == n
        assert Bits.number_at(bin, t, 1) == zero
        assert Bits.number_at(bin, t, 2) == one
      end
    end
  end
end
