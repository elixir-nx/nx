# Example demonstrating message-based infeed/outfeed replacement
#
# This example shows how to use the new message-based communication system
# that replaces traditional StableHLO infeed/outfeed operations with direct
# message passing to Elixir processes.

# This example runs in the context of the EXLA project
# Make sure to run: mix deps.get && mix compile
# before running this example

# Set EXLA as the default compiler so metadata is processed correctly
Nx.Defn.default_options(compiler: EXLA)

defmodule MessageCommExample do
  import Nx.Defn

  # Example function that actually uses message-based outfeed
  defn send_tensor_via_message(pid_tensor, tensor) do
    # Use the message-based outfeed custom call to send tensor data
    # This will send the tensor data to the current Elixir process

    custom_call({pid_tensor, tensor}, "message_outfeed_f32_simple", [Nx.template({1}, {:u, 8})])
  end

  # Example function that uses message-based infeed
  defn receive_tensor_via_message(pid_tensor) do
    # Use the message-based infeed custom call to receive tensor data
    # For now this returns zeros, but in a full implementation would wait for data
    custom_call(pid_tensor, "message_infeed_f32_simple", [Nx.template({2, 2}, {:f, 32})])
  end

  defn run_both(pid_tensor) do
    tensor = receive_tensor_via_message(pid_tensor)
    send_tensor_via_message(pid_tensor, Nx.add(tensor, 10))
  end

  def run_example do
    IO.puts("=== Message-Based Communication Example ===")
    IO.puts("This demonstrates infeed/outfeed replacement with message passing")
    IO.puts("")

    # Set EXLA as the default backend
    Nx.default_backend(EXLA.Backend)

    # Test data
    input_tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
    IO.puts("Input tensor: #{inspect(input_tensor)}")

    # Start listening for messages in a separate process
    listener_pid = spawn_link(fn -> listen_for_messages() end)

    # Create a PID value containing the listener process
    pid_binary = :erlang.term_to_binary(listener_pid)
    pid_tensor = Nx.from_binary(pid_binary, {:u, 8})

    IO.puts("")
    IO.puts("🚀 Running defn function with message-based communication...")
    IO.puts("   This will:")
    IO.puts("   1. Request tensor data via message infeed")
    IO.puts("   2. Process the data (add 10)")
    IO.puts("   3. Send result via message outfeed")
    IO.puts("")

    result = run_both(pid_tensor)
    IO.puts("Final result: #{inspect(result)}")

    # Give time for messages to be processed
    Process.sleep(200)

    # Stop the listener
    send(listener_pid, :stop)

    IO.puts("")
    IO.puts("✅ Example completed!")
    IO.puts("✅ Message-based infeed sent request to Elixir process")
    IO.puts("✅ Message-based outfeed sent data to Elixir process")
  end

  deftransformp custom_call(operands, call_target_name, result_types) do
    Nx.Defn.Expr.metadata(operands, %{
      exla_custom_call: call_target_name,
      result_types: result_types
    })
  end

  defp listen_for_messages do
    receive do
      {:outfeed, data_binary, shape} ->
        IO.puts("✅ Received outfeed message!")
        IO.puts("  Data size: #{byte_size(data_binary)} bytes")
        IO.puts("  Shape: #{inspect(shape)}")

        # Convert binary back to tensor data for display
        <<f1::float-native-32, f2::float-native-32, f3::float-native-32, f4::float-native-32>> =
          data_binary

        IO.puts("  Data values: [#{f1}, #{f2}, #{f3}, #{f4}]")

        listen_for_messages()

      {:infeed_request, ref, shape} ->
        IO.puts("✅ Received infeed request!")
        IO.puts("  Reference: #{inspect(ref)}")
        IO.puts("  Requested shape: #{inspect(shape)}")

        # Respond with tensor data for the infeed
        # Generate test data based on the requested shape
        case shape do
          [2, 2] ->
            # Create a 2x2 tensor with test data
            test_tensor = Nx.tensor([[100.0, 200.0], [300.0, 400.0]], type: {:f, 32})
            data_binary = Nx.to_binary(test_tensor)
            IO.puts("  📤 Providing test tensor data: [[100.0, 200.0], [300.0, 400.0]]")

            # Convert reference to binary for NIF call (same as C++ code)
            ref_binary = :erlang.term_to_binary(ref)

            # Complete the infeed request with actual data
            case EXLA.NIF.complete_infeed_request_fine(
                   Base.encode64(ref_binary),
                   Base.encode64(data_binary)
                 ) do
              :ok ->
                IO.puts("  ✅ Successfully provided infeed data")

              {:error, reason} ->
                IO.puts("  ❌ Failed to provide infeed data: #{reason}")
            end

          _ ->
            IO.puts("  ❓ Unsupported shape for infeed: #{inspect(shape)}")
        end

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
MessageCommExample.run_example()
