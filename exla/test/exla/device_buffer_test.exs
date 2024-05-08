defmodule EXLA.DeviceBufferTest do
  use ExUnit.Case, async: true

  alias EXLA.{DeviceBuffer, Typespec}

  import EXLAHelpers

  describe "buffer" do
    test "place_on_device/4" do
      b1 = DeviceBuffer.place_on_device(<<1::32>>, Typespec.tensor({:s, 32}, {}), client(), 0)
      assert is_reference(b1.ref)
    end

    test "read/2" do
      b1 =
        DeviceBuffer.place_on_device(
          <<1::32, 2::32, 3::32, 4::32>>,
          Typespec.tensor({:s, 32}, {4}),
          client(),
          0
        )

      # non-destructive
      assert <<1::32, 2::32, 3::32, 4::32>> == DeviceBuffer.read(b1)
      assert <<1::32, 2::32, 3::32, 4::32>> == DeviceBuffer.read(b1)

      # doesn't change after deallocation
      binary = DeviceBuffer.read(b1)
      :ok = DeviceBuffer.deallocate(b1)
      assert binary == <<1::32, 2::32, 3::32, 4::32>>

      assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
        DeviceBuffer.read(b1)
      end
    end

    test "deallocate/1" do
      b1 = DeviceBuffer.place_on_device(<<1::32>>, Typespec.tensor({:s, 32}, {}), client(), 0)

      assert :ok = DeviceBuffer.deallocate(b1)
      assert :already_deallocated = DeviceBuffer.deallocate(b1)
    end

    test "shape regression" do
      # This isn't really a valid buffer we're creating. This test is only meant to check that
      # we don't get an "unable to get shape" error when any of the dimensions exceeds max int32
      max_int32 = 2_147_483_647
      typespec = %EXLA.Typespec{type: {:s, 64}, shape: {max_int32 + 1}}
      device_id = 0
      client = EXLA.Client.fetch!(:host)

      # this assertion seems pointless, but it's just so we ensure this doesn't raise
      assert %DeviceBuffer{} =
               DeviceBuffer.place_on_device(<<0::64>>, typespec, client, device_id)
    end
  end
end
