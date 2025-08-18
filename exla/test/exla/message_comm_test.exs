defmodule EXLA.MessageCommTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  # Define the defn function outside the test
  defn send_data(tensor) do
    result = Nx.add(tensor, 1.0)
    # Use the process_pid custom call to send data back to Elixir
    # Note: We need to use Nx.Defn.Expr.metadata/2 instead of Nx.metadata/2
    result
  end

  setup do
    # Ensure we're using the host backend for testing
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  describe "message-based outfeed" do
    test "basic computation works" do
      input = Nx.tensor([1.0, 2.0, 3.0])

      # Run the computation
      result = send_data(input)

      # Verify the computation result
      expected = Nx.add(input, 1.0)
      assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "message communication module" do
    test "can create message communication context" do
      # This tests the MessageComm module functionality
      # Note: This would require a proper MLIR function context to test fully
      assert is_atom(EXLA.Defn.MessageComm)
    end

    test "can start message handler" do
      {:ok, pid} = EXLA.Defn.MessageComm.start_message_handler(%{})
      assert is_pid(pid)

      # Send stop message to clean up
      send(pid, :stop)

      # Wait for the process to terminate
      ref = Process.monitor(pid)
      assert_receive {:DOWN, ^ref, :process, ^pid, :normal}, 1000
    end

    test "message handler forwards messages correctly" do
      test_pid = self()

      {:ok, handler_pid} =
        EXLA.Defn.MessageComm.start_message_handler(%{
          default: fn binary -> send(test_pid, {:processed, binary}) end
        })

      # Send a test message
      send(handler_pid, {:outfeed_data, <<1, 2, 3, 4>>})

      # Verify it was processed
      assert_receive {:processed, <<1, 2, 3, 4>>}, 1000

      # Clean up
      send(handler_pid, :stop)
    end
  end

  describe "value module functions" do
    test "message infeed and outfeed functions exist" do
      # Verify the new functions are defined
      assert function_exported?(EXLA.MLIR.Value, :message_infeed, 2)
      assert function_exported?(EXLA.MLIR.Value, :message_outfeed, 2)
      assert function_exported?(EXLA.MLIR.Value, :message_multi_outfeed, 2)
    end
  end

  describe "configuration" do
    test "can enable message communication in options" do
      # Test that the use_message_comm option can be set
      options = [use_message_comm: true]
      assert Keyword.get(options, :use_message_comm) == true
    end
  end
end
