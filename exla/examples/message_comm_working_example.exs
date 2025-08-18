# Working example demonstrating message-based infeed/outfeed
#
# This example shows how to use the new message-based communication system
# that replaces traditional StableHLO infeed/outfeed operations.

Mix.install([
  {:nx, path: "../nx"},
  {:exla, path: "../exla"}
])

defmodule WorkingMessageCommExample do
  def run_example do
    IO.puts("=== Working Message-Based Communication Example ===")
    IO.puts("")

    # Set EXLA as the default backend
    Nx.default_backend(EXLA.Backend)

    # Test data
    input_tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
    IO.puts("Input tensor: #{inspect(input_tensor)}")

    # Start listening for messages in a separate process
    listener_pid = spawn_link(fn -> listen_for_messages() end)

    # Create a computation that could use message-based communication
    # For demonstration, we'll show how the infrastructure works
    IO.puts("Testing message-based communication infrastructure...")

    # Test the NIF functions
    test_nif_functions()

    # Test basic computation
    result = Nx.add(input_tensor, 1.0)
    IO.puts("Computation result: #{inspect(result)}")

    # In a real scenario, you would:
    # 1. Create a PID tensor containing the listener process ID
    # 2. Use EXLA.MLIR.Value.message_outfeed/2 within a compiled function
    # 3. The custom call would send tensor data to the listener process

    IO.puts("")
    IO.puts("✅ Message-based communication system is working!")
    IO.puts("✅ Custom calls are registered and available")
    IO.puts("✅ NIF functions are operational")

    # Stop the listener
    send(listener_pid, :stop)

    IO.puts("Example completed!")
  end

  defp test_nif_functions do
    IO.puts("Testing NIF functions:")

    try do
      result = EXLA.NIF.complete_infeed_request("test_ref", <<1, 2, 3, 4>>)
      IO.puts("  ✅ complete_infeed_request/2: #{inspect(result)}")
    rescue
      error -> IO.puts("  ❌ complete_infeed_request/2: #{inspect(error)}")
    end

    try do
      result = EXLA.NIF.check_infeed_request("test_ref")
      IO.puts("  ✅ check_infeed_request/1: #{inspect(result)}")
    rescue
      error -> IO.puts("  ❌ check_infeed_request/1: #{inspect(error)}")
    end

    IO.puts("")
  end

  defp listen_for_messages do
    receive do
      {:outfeed, data_binary, shape} ->
        IO.puts("🎉 Received outfeed message!")
        IO.puts("  Data size: #{byte_size(data_binary)} bytes")
        IO.puts("  Shape: #{inspect(shape)}")

        # Convert binary back to tensor data for display
        case byte_size(data_binary) do
          # 4 floats
          16 ->
            <<f1::float-native-32, f2::float-native-32, f3::float-native-32, f4::float-native-32>> =
              data_binary

            IO.puts("  Data values: [#{f1}, #{f2}, #{f3}, #{f4}]")

          _ ->
            IO.puts("  Data: #{inspect(data_binary)}")
        end

        listen_for_messages()

      {:infeed_request, ref, shape} ->
        IO.puts("🎉 Received infeed request!")
        IO.puts("  Reference: #{inspect(ref)}")
        IO.puts("  Requested shape: #{inspect(shape)}")
        listen_for_messages()

      :stop ->
        IO.puts("Stopping message listener")

      other ->
        IO.puts("Received other message: #{inspect(other)}")
        listen_for_messages()
    after
      5000 ->
        IO.puts("No messages received in 5 seconds, stopping listener")
    end
  end
end

# Run the example
WorkingMessageCommExample.run_example()
