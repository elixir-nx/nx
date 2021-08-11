defmodule EXLA.DefnAPITest do
  use ExUnit.Case, async: true

  import Nx.Defn

  describe "options" do
    @defn_compiler {EXLA, run_options: [keep_on_device: true]}
    defn add_two_keep_on_device(a, b), do: a + b

    # Ignored logged errors, since they are expected
    @tag capture_log: true
    test "keeps data on device" do
      tensor = add_two_keep_on_device(1, 2)
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert %EXLA.DeviceBackend{state: {ref, :default}} = tensor.data
      assert is_reference(ref)

      assert tensor |> Nx.backend_transfer() |> Nx.to_binary() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   "Invalid buffer passed to Execute() as argument 1 to replica 0: Invalid argument: Buffer has been deleted or donated.",
                   fn -> add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor) end

      assert_raise RuntimeError,
                   "CopyToHostAsync() called on deleted or donated buffer",
                   fn -> Nx.backend_transfer(tensor) end
    end
  end

  describe "stream" do
    defn defn_sum(entry, acc), do: {acc, entry + acc}

    test "poc" do
      alias EXLA.{Buffer, Shape}
      [res] = EXLA.stream(&defn_sum/2, [0, 0])

      yes = %Buffer{data: <<1::8-native>>, shape: Shape.make_shape({:pred, 8}, {})}
      no = %Buffer{data: <<0::8-native>>, shape: Shape.make_shape({:pred, 8}, {})}

      one = %Buffer{data: <<1::64-native>>, shape: Shape.make_shape({:s, 64}, {})}
      two = %Buffer{data: <<2::64-native>>, shape: Shape.make_shape({:s, 64}, {})}

      client = EXLA.Client.fetch!(:default)
      assert :ok = Buffer.to_infeed(yes, client, 0)
      assert :ok = Buffer.to_infeed(one, client, 0)

      assert %Buffer{data: <<0::64-native>>} =
               Buffer.from_outfeed(client, 0, Shape.make_shape({:s, 64}, {}))

      assert :ok = Buffer.to_infeed(yes, client, 0)
      assert :ok = Buffer.to_infeed(two, client, 0)

      assert %Buffer{data: <<1::64-native>>} =
               Buffer.from_outfeed(client, 0, Shape.make_shape({:s, 64}, {}))

      assert :ok = Buffer.to_infeed(no, client, 0)
      assert <<3::64-native>> == Buffer.read(res.ref)
    end
  end
end
