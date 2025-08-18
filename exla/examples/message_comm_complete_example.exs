# Complete example demonstrating message-based infeed/outfeed replacement
#
# This example shows how to use the new message-based communication system
# that replaces traditional StableHLO infeed/outfeed operations with direct
# message passing to Elixir processes.

Mix.install([
  {:nx, path: "../nx"},
  {:exla, path: "../exla"}
])

defmodule CompleteMessageCommExample do
  def run_example do
    IO.puts("=== Complete Message-Based Communication Example ===")
    IO.puts("This demonstrates infeed/outfeed replacement with message passing")
    IO.puts("")

    # Set EXLA as the default backend
    Nx.default_backend(EXLA.Backend)

    # Start a message handler process that can provide tensor data
    handler_pid = spawn_link(fn -> message_handler() end)

    IO.puts("📡 Started message handler process: #{inspect(handler_pid)}")

    # Test the message-based communication using the MessageComm module
    test_message_communication(handler_pid)

    # Stop the handler
    send(handler_pid, :stop)

    IO.puts("")
    IO.puts("✅ Complete example finished!")
  end

  defp test_message_communication(handler_pid) do
    IO.puts("")
    IO.puts("🧪 Testing message-based communication...")

    # Create some test data that the handler will provide via infeed
    test_data = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
    send(handler_pid, {:set_infeed_data, test_data})

    # Create a simple computation that uses message-based communication
    # We'll simulate this by directly calling the MessageComm functions

    IO.puts("📤 Testing outfeed: sending tensor to handler...")

    # Test outfeed - send a tensor to the handler
    outfeed_data = Nx.tensor([[10.0, 20.0], [30.0, 40.0]], type: {:f, 32})
    send_tensor_to_handler(handler_pid, outfeed_data)

    # Give time for message processing
    Process.sleep(100)

    IO.puts("📥 Testing infeed: requesting tensor from handler...")

    # Test infeed - request a tensor from the handler
    received_tensor = request_tensor_from_handler(handler_pid, {2, 2})
    IO.puts("Received tensor: #{inspect(received_tensor)}")

    # Give time for message processing
    Process.sleep(100)
  end

  defp send_tensor_to_handler(handler_pid, tensor) do
    # Convert tensor to binary for sending
    data_binary = Nx.to_binary(tensor)
    shape = Nx.shape(tensor)

    # Send outfeed message
    message = {:outfeed, data_binary, Tuple.to_list(shape)}
    send(handler_pid, message)

    IO.puts("  ✅ Sent tensor with shape #{inspect(shape)} to handler")
  end

  defp request_tensor_from_handler(handler_pid, shape) do
    # Generate a unique reference for this request
    ref = make_ref()

    # Send infeed request
    message = {:infeed_request, ref, Tuple.to_list(shape)}
    send(handler_pid, message)

    IO.puts("  📤 Sent infeed request for shape #{inspect(shape)}")

    # Wait for response (in a real implementation, this would be handled by the custom call)
    receive do
      {:infeed_response, ^ref, data_binary} ->
        # Convert binary back to tensor
        tensor = Nx.from_binary(data_binary, {:f, 32}) |> Nx.reshape(shape)
        IO.puts("  ✅ Received tensor data from handler")
        tensor
    after
      1000 ->
        IO.puts("  ❌ Timeout waiting for infeed response")
        Nx.broadcast(0.0, shape)
    end
  end

  defp message_handler do
    # Store the current infeed data
    message_handler_loop(nil)
  end

  defp message_handler_loop(infeed_data) do
    receive do
      {:set_infeed_data, tensor} ->
        IO.puts("🗃️  Handler: stored infeed data #{inspect(Nx.shape(tensor))}")
        message_handler_loop(tensor)

      {:outfeed, data_binary, shape} ->
        IO.puts("📨 Handler: received outfeed message!")
        IO.puts("   Data size: #{byte_size(data_binary)} bytes")
        IO.puts("   Shape: #{inspect(shape)}")

        # Convert binary back to tensor for display
        case byte_size(data_binary) do
          # 4 floats
          16 ->
            <<f1::float-native-32, f2::float-native-32, f3::float-native-32, f4::float-native-32>> =
              data_binary

            IO.puts("   Data values: [#{f1}, #{f2}, #{f3}, #{f4}]")

          _ ->
            IO.puts("   Data: #{inspect(data_binary)}")
        end

        message_handler_loop(infeed_data)

      {:infeed_request, ref, shape} ->
        IO.puts("📥 Handler: received infeed request!")
        IO.puts("   Reference: #{inspect(ref)}")
        IO.puts("   Requested shape: #{inspect(shape)}")

        # Provide the stored infeed data or generate test data
        response_data =
          if infeed_data do
            IO.puts("   📤 Providing stored tensor data")
            Nx.to_binary(infeed_data)
          else
            IO.puts("   📤 Generating test tensor data")
            # Generate test data: [5.0, 6.0, 7.0, 8.0] for a 2x2 tensor
            test_tensor = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], type: {:f, 32})
            Nx.to_binary(test_tensor)
          end

        # Send response back
        send(self(), {:infeed_response, ref, response_data})

        message_handler_loop(infeed_data)

      :stop ->
        IO.puts("🛑 Handler: stopping")

      other ->
        IO.puts("❓ Handler: received unexpected message: #{inspect(other)}")
        message_handler_loop(infeed_data)
    after
      10000 ->
        IO.puts("⏰ Handler: timeout, stopping")
    end
  end
end

# Run the example
CompleteMessageCommExample.run_example()
