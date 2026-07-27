defmodule EXLA.Defn.DonationTest do
  use EXLA.Case, async: true

  alias EXLA.DeviceBuffer

  defp on_device(value) do
    Nx.backend_transfer(Nx.tensor(value), {EXLA.Backend, client: :host})
  end

  describe "Nx.donate/1" do
    test "donates a single argument and consumes its buffer" do
      x = Nx.donate(on_device([1, 2, 3, 4]))
      %EXLA.Backend{buffer: %DeviceBuffer{} = orig} = x.data

      fun = EXLA.jit(&Nx.add(&1, 1))
      result = fun.(x)

      assert Nx.to_flat_list(result) == [2, 3, 4, 5]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(orig)
      end
    end

    test "donates both args of a two-arg function" do
      x = Nx.donate(on_device([1, 2, 3]))
      y = Nx.donate(on_device([10, 20, 30]))
      %EXLA.Backend{buffer: %DeviceBuffer{} = xb} = x.data
      %EXLA.Backend{buffer: %DeviceBuffer{} = yb} = y.data

      fun = EXLA.jit(&{Nx.add(&1, &2), Nx.subtract(&1, &2)})
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

    test "donating a composite argument consumes every marked leaf" do
      a = Nx.donate(on_device([1, 2]))
      b = Nx.donate(on_device([3, 4]))
      %EXLA.Backend{buffer: %DeviceBuffer{} = ab} = a.data
      %EXLA.Backend{buffer: %DeviceBuffer{} = bb} = b.data

      fun = EXLA.jit(fn {l, r} -> {Nx.add(l, 1), Nx.multiply(r, 2)} end)
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
      x = Nx.donate(on_device([1, 2, 3]))
      y = on_device([10, 20, 30])
      %EXLA.Backend{buffer: %DeviceBuffer{} = yb} = y.data

      fun = EXLA.jit(&Nx.add(&1, &2))
      result = fun.(x, y)

      assert Nx.to_flat_list(result) == [11, 22, 33]
      # `y` was not donated; reading should still succeed.
      assert byte_size(DeviceBuffer.read(yb)) > 0
    end

    test "raises when no output has a matching shape/dtype" do
      assert_raise ArgumentError, ~r"no output with matching shape", fn ->
        EXLA.jit(&Nx.sum/1).(Nx.donate(on_device([1, 2, 3, 4])))
      end
    end

    test "different donate sets produce distinct cached executables" do
      # Same shape/typespec, but distinct donation must not share the executable.
      x = on_device([1, 2, 3])

      _ = EXLA.jit(&Nx.add(&1, 1)).(x)
      # If this cached the non-donating executable, the buffer wouldn't be consumed below.
      x2 = Nx.donate(on_device([1, 2, 3]))
      %EXLA.Backend{buffer: %DeviceBuffer{} = xb2} = x2.data

      _ = EXLA.jit(&Nx.add(&1, 1)).(x2)

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(xb2)
      end
    end

    test "donates only the marked leaves of a map" do
      a = Nx.donate(on_device([1, 2]))
      b = on_device([3, 4])
      %EXLA.Backend{buffer: %DeviceBuffer{} = ab} = a.data
      %EXLA.Backend{buffer: %DeviceBuffer{} = bb} = b.data

      fun = EXLA.jit(fn %{x: x, y: y} -> %{x: Nx.add(x, 1), y: Nx.multiply(y, 2)} end)
      %{x: x, y: y} = fun.(%{x: a, y: b})

      assert Nx.to_flat_list(x) == [2, 3]
      assert Nx.to_flat_list(y) == [6, 8]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(ab)
      end

      assert byte_size(DeviceBuffer.read(bb)) > 0
    end

    test "raises when donation is combined with sharded execution" do
      mesh = %Nx.Mesh{name: "mesh", shape: {1}}

      assert_raise ArgumentError, ~r"not currently supported with sharded execution", fn ->
        EXLA.shard_jit(&Nx.add(&1, 1), mesh, input_shardings: [%{}]).([
          [Nx.donate(on_device([1, 2, 3]))]
        ])
      end
    end

    test "compile bakes donation from donatable templates" do
      template = Nx.donate(Nx.template({3}, {:s, 32}))
      fun = EXLA.compile(&Nx.add(&1, 1), [template])

      x = on_device([1, 2, 3])
      %EXLA.Backend{buffer: %DeviceBuffer{} = xb} = x.data

      assert Nx.to_flat_list(fun.(x)) == [2, 3, 4]

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(xb)
      end
    end
  end
end
