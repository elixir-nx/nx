defmodule Nx.BinaryBackend.BinaryTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.Binary

  doctest Binary

  import Nx.Type, only: [is_float_type: 1, is_integer_type: 1]

  @float_types [{:f, 64}, {:f, 32}, {:bf, 16}]
  @int_types [{:s, 64}, {:s, 32}, {:u, 64}, {:u, 32}, {:u, 16}]

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

  defp rand_num(t) when is_integer_type(t) do
    rand_int(t)
  end

  defp rand_num(t) when is_float_type(t) do
    rand_float(t)
  end

  defp rand_num_encodable(t) do
    t
    |> rand_num()
    |> Binary.from_number(t)
    |> Binary.to_number(t)
  end

  defp min_val({:u, _}), do: 0
  defp min_val(_), do: -10_000

  defp max_val(_), do: 10_000

  defp zero(t), do: 0 * one(t)

  defp one(t) when is_float_type(t), do: 1.0
  defp one(t) when is_integer_type(t), do: 1

  defp to_type(n, t) when is_integer_type(t) and is_float(n) do
    n
    |> Float.round()
    |> trunc()
  end

  defp to_type(n, t) when is_integer_type(t) and is_integer(n) do
    n
  end

  defp to_type(n, t) when is_float_type(t) do
    n * 1.0
  end

  describe "from_number/2 and to_number/2" do
    test "can encode and decode all float types" do
      for t <- @float_types do
        n1 = rand_float()
        b = Binary.from_number(n1, t)
        n2 = Binary.to_number(b, t)
        eps = epsilon(t)
        assert_in_delta(
          n1,
          n2,
          eps,
          "type #{inspect(t)} was significantly different (eps: #{eps}) - before: #{n1} - after: #{n2}"
        )
      end
    end

    test "can encode and decode all integer types" do
      for t <- @int_types do
        n1 = rand_int(t)
        b = Binary.from_number(n1, t)
        n2 = Binary.to_number(b, t)
        assert n1 == n2, "int type #{inspect(t)} encode/decode failed - before: #{n1} - after: #{n2}"
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
          |> Enum.map(fn x -> Binary.from_number(x, t) end)
          |> IO.iodata_to_binary()
        
        assert Binary.number_at(bin, t, 0) == n
        assert Binary.number_at(bin, t, 1) == zero
        assert Binary.number_at(bin, t, 2) == one
      end
    end
  end

  describe "number_at/4" do
    test "works for all types with rank 1 tensor" do
      for type <- @int_types ++ @float_types do
        tensor = Nx.iota({10}, type: type)
        shape = Nx.shape(tensor)
        assert tensor.data.__struct__ == Nx.BinaryBackend
        data = tensor.data.state
        for i <- 0..9 do
          assert Binary.number_at(data, shape, type, {i}) == to_type(i, type)
        end
      end
    end

    test "works for all types with rank 2 tensor" do
      for type <- @int_types ++ @float_types do
        tensor = Nx.iota({10, 10}, type: type)
        shape = Nx.shape(tensor)
        assert tensor.data.__struct__ == Nx.BinaryBackend
        data = tensor.data.state
        for i <- 0..99 do
          i1 = div(i, 10)
          i2 = rem(i, 10)
          assert Binary.number_at(data, shape, type, {i1, i2}) == to_type(i, type)
        end
      end
    end

    test "works for all types with rank 3 tensor" do
      for type <- @int_types ++ @float_types do
        tensor = Nx.iota({3, 3, 3}, type: type)
        shape = Nx.shape(tensor)
        assert tensor.data.__struct__ == Nx.BinaryBackend
        data = tensor.data.state
        for i <- 0..26 do
          i1 = div(i, 9)
          i1rem = rem(i, 9)
          i2 = div(i1rem, 3)
          i3 = rem(i, 3)
          idx_tup = {i1, i2, i3}
          got = Binary.number_at(data, shape, type, idx_tup)
          expected = to_type(i, type)
          assert got == expected, """
          Binary.number_at/4 failed at index #{i} with type #{inspect(type)} and index_tuple #{inspect(idx_tup)}
            expected: #{expected}
            got: #{got}
          """
        end
      end
    end
  end
end