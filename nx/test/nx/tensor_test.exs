defmodule Nx.TensorTest do
  use ExUnit.Case, async: true

  describe "backend_transfer" do
    test "transfers existing tensor" do
      Nx.tensor([1, 2, 3]) |> Nx.backend_transfer({ProcessBackend, key: :example})
      assert Process.get(:example) == <<1::64-native, 2::64-native, 3::64-native>>
    end
  end

  describe "backend_copy" do
    test "copies existing tensor" do
      Nx.tensor([1, 2, 3]) |> Nx.backend_copy({ProcessBackend, key: :example})
      assert Process.get(:example) == <<1::64-native, 2::64-native, 3::64-native>>
    end
  end

  describe "backend_deallocate" do
    test "deallocates existing tensor" do
      t = Nx.tensor([1, 2, 3]) |> Nx.backend_transfer({ProcessBackend, key: :example})
      assert Process.get(:example) == <<1::64-native, 2::64-native, 3::64-native>>
      assert Nx.backend_deallocate(t) == :ok
      refute Process.get(:example)
      assert Nx.backend_deallocate(t) == :already_deallocated
    end
  end

  describe "tuples" do
    test "on backend_transfer" do
      assert Nx.backend_transfer({Nx.tensor(1), 2}) == {Nx.tensor(1), Nx.tensor(2)}
    end

    test "on backend_copy" do
      assert Nx.backend_copy({Nx.tensor(1), 2}) == {Nx.tensor(1), Nx.tensor(2)}
    end

    test "on backend_deallocate" do
      assert Nx.backend_deallocate({Nx.tensor(1), 2}) == :ok
    end
  end

  describe "maps" do
    test "on backend_transfer" do
      assert Nx.backend_transfer(%{foo: Nx.tensor(1), bar: 2}) == %{
               foo: Nx.tensor(1),
               bar: Nx.tensor(2)
             }
    end

    test "on backend_copy" do
      assert Nx.backend_copy(%{foo: Nx.tensor(1), bar: 2}) == %{
               foo: Nx.tensor(1),
               bar: Nx.tensor(2)
             }
    end

    test "on backend_deallocate" do
      assert Nx.backend_deallocate(%{foo: Nx.tensor(1), bar: 2}) == :ok
    end
  end

  describe "default backend" do
    setup do
      Nx.default_backend({ProcessBackend, key: :example})
      :ok
    end

    test "on tensor" do
      Nx.tensor([1, 2, 3])
      assert Process.get(:example)
    end

    test "on from_binary" do
      Nx.from_binary(<<1, 2, 3>>, {:u, 8})
      assert Process.get(:example)
    end

    test "on eye" do
      assert_raise RuntimeError, "not supported", fn -> Nx.eye(3) end
    end

    test "on iota" do
      assert_raise RuntimeError, "not supported", fn -> Nx.iota({2, 2}) end
    end
  end

  describe "access" do
    test "works with stepped ranges" do
      assert Nx.iota({2, 3, 5})[[.., 0..-1//3, 0..-1//2]] ==
               Nx.tensor([
                 [
                   [0, 2, 4]
                 ],
                 [
                   [15, 17, 19]
                 ]
               ])

      assert Nx.iota({2, 3, 5})[[.., 0..-1//2, 0..-1//2]] ==
               Nx.tensor([
                 [
                   [0, 2, 4],
                   [10, 12, 14]
                 ],
                 [
                   [15, 17, 19],
                   [25, 27, 29]
                 ]
               ])
    end

    test "works with negative steps" do
      iota = Nx.iota({2, 5})

      for step <- -5..-1 do
        assert iota[[0, 4..0//step]] == Nx.tensor(Enum.to_list(4..0//step))
        assert iota[[1, 4..0//step]] == Nx.tensor(Enum.to_list(9..5//step))
      end
    end
  end
end
