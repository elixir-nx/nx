defmodule EXLA.FFI.StreamTest do
  use ExUnit.Case, async: true

  alias EXLA.FFI.Stream

  describe "basic queue semantics" do
    test "start_link registers process by name" do
      name = :exla_ffi_stream_test_start
      assert {:ok, pid} = Stream.start_link(name: name)
      assert Process.whereis(name) == pid
    end

    test "push_infeed and pop_outfeed are independent" do
      name = :exla_ffi_stream_test_queues
      {:ok, _pid} = Stream.start_link(name: name)

      # For now, pushing infeed does not affect outfeed
      :ok = Stream.push_infeed(name, :foo)
      assert :empty == Stream.pop_outfeed(name, 10)
    end
  end
end
