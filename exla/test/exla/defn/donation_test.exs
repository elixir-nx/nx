defmodule EXLA.Defn.DonationTest do
  use EXLA.Case, async: true

  alias EXLA.DeviceBuffer

  defp on_device(value) do
    Nx.backend_transfer(Nx.tensor(value), {EXLA.Backend, client: :host})
  end

  describe "donate_argnums" do
    test "donates a single positional argument and consumes its buffer" do
      x = on_device([1, 2, 3, 4])
      %EXLA.Backend{buffer: %DeviceBuffer{} = orig} = x.data

      fun = EXLA.jit(&Nx.add(&1, 1), donate_argnums: [0])
      result = fun.(x)

      assert Nx.to_flat_list(result) == [2, 3, 4, 5]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(orig)
      end
    end

    test "donates both args of a two-arg function" do
      x = on_device([1, 2, 3])
      y = on_device([10, 20, 30])
      %EXLA.Backend{buffer: %DeviceBuffer{} = xb} = x.data
      %EXLA.Backend{buffer: %DeviceBuffer{} = yb} = y.data

      fun = EXLA.jit(&{Nx.add(&1, &2), Nx.subtract(&1, &2)}, donate_argnums: [0, 1])
      {sum, diff} = fun.(x, y)

      assert Nx.to_flat_list(sum) == [11, 22, 33]
      assert Nx.to_flat_list(diff) == [-9, -18, -27]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(xb)
      end

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(yb)
      end
    end

    test "donating a composite argument consumes every leaf within it" do
      a = on_device([1, 2])
      b = on_device([3, 4])
      %EXLA.Backend{buffer: %DeviceBuffer{} = ab} = a.data
      %EXLA.Backend{buffer: %DeviceBuffer{} = bb} = b.data

      fun = EXLA.jit(fn {l, r} -> {Nx.add(l, 1), Nx.multiply(r, 2)} end, donate_argnums: [0])
      {l, r} = fun.({a, b})

      assert Nx.to_flat_list(l) == [2, 3]
      assert Nx.to_flat_list(r) == [6, 8]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(ab)
      end

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(bb)
      end
    end

    test "donation does not consume non-donated args" do
      x = on_device([1, 2, 3])
      y = on_device([10, 20, 30])
      %EXLA.Backend{buffer: %DeviceBuffer{} = yb} = y.data

      fun = EXLA.jit(&Nx.add(&1, &2), donate_argnums: [0])
      result = fun.(x, y)

      assert Nx.to_flat_list(result) == [11, 22, 33]
      # `y` was not donated; reading should still succeed.
      assert byte_size(DeviceBuffer.read(yb)) > 0
    end

    test "raises when no output has a matching shape/dtype" do
      assert_raise ArgumentError, ~r"no output with matching shape", fn ->
        EXLA.jit(&Nx.sum/1, donate_argnums: [0]).(on_device([1, 2, 3, 4]))
      end
    end

    test "raises when an argnum is out of range" do
      assert_raise ArgumentError, ~r":donate_argnums entries must be in the range", fn ->
        EXLA.jit(&Nx.add(&1, 1), donate_argnums: [5]).(on_device([1, 2, 3]))
      end
    end

    test "raises when :donate_argnums is malformed" do
      assert_raise ArgumentError, ~r":donate_argnums must be a list", fn ->
        EXLA.jit(&Nx.add(&1, 1), donate_argnums: [-1]).(on_device([1, 2, 3]))
      end

      assert_raise ArgumentError, ~r":donate_argnums must be a list", fn ->
        EXLA.jit(&Nx.add(&1, 1), donate_argnums: :foo).(on_device([1, 2, 3]))
      end
    end

    test "duplicates in :donate_argnums are deduped silently" do
      x = on_device([1, 2, 3])
      %EXLA.Backend{buffer: %DeviceBuffer{} = xb} = x.data

      fun = EXLA.jit(&Nx.add(&1, 1), donate_argnums: [0, 0])
      assert Nx.to_flat_list(fun.(x)) == [2, 3, 4]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(xb)
      end
    end

    test "different donate sets produce distinct cached executables" do
      # Same shape/typespec, but distinct option values must not share the executable.
      x = on_device([1, 2, 3])

      _ = EXLA.jit(&Nx.add(&1, 1)).(x)
      # If this cached the non-donating executable, the buffer wouldn't be consumed below.
      x2 = on_device([1, 2, 3])
      %EXLA.Backend{buffer: %DeviceBuffer{} = xb2} = x2.data

      _ = EXLA.jit(&Nx.add(&1, 1), donate_argnums: [0]).(x2)

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(xb2)
      end
    end
  end
end
