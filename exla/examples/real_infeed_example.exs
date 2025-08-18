# Real infeed example using the message-based custom calls
#
# This demonstrates actual infeed functionality where a defn function
# requests tensor data from an Elixir process and receives real data back.

# Set EXLA as the default compiler so metadata is processed correctly
Nx.Defn.default_options(compiler: EXLA)

defmodule RealInfeedExample do
  import Nx.Defn

  # Define a function that uses message-based infeed
  defn compute_with_infeed(pid_tensor) do
    # Request a 2x2 tensor via infeed
    # This will send a message to the Elixir process and wait for data
    infeed_data = request_tensor_data(pid_tensor, {2, 2})

    # Process the received data - ensure we return the right shape
    result = Nx.add(infeed_data, 10.0)

    result
  end

  # This is the actual infeed custom call using metadata
  deftransformp request_tensor_data(pid_tensor, shape) do
    # Use the message-based infeed custom call to receive tensor data
    # This will send an infeed request to the Elixir process and wait for data
    result_template =
      case shape do
        {2, 2} -> Nx.template({2, 2}, {:f, 32})
        _ -> Nx.template({4}, {:f, 32})
      end

    custom_call(pid_tensor, "message_infeed_f32_simple", [result_template])
  end

  # Add the custom_call transform function like in message_comm_example.exs
  deftransformp custom_call(operands, call_target_name, [result_type]) do
    %{
      Nx.Defn.Expr.metadata(operands, %{
        exla_custom_call: call_target_name,
        result_types: [result_type]
      })
      | shape: result_type.shape,
        names: result_type.names,
        type: result_type.type
    }
  end

  def run_example do
    IO.puts("=== Real Message-Based Infeed Example ===")
    IO.puts("")

    # Set EXLA as the default backend
    Nx.default_backend(EXLA.Backend)

    # Start a message handler process
    handler_pid = spawn_link(fn -> infeed_handler() end)
    IO.puts("🎯 Started infeed handler: #{inspect(handler_pid)}")

    # Create PID tensor for the handler process
    pid_binary = :erlang.term_to_binary(handler_pid)
    pid_tensor = Nx.from_binary(pid_binary, {:u, 8})
    IO.puts("📡 Created PID tensor: #{inspect(Nx.shape(pid_tensor))} elements")

    # Run the computation
    IO.puts("🚀 Running defn function with message-based infeed...")
    result = compute_with_infeed(pid_tensor)

    IO.puts("📊 Result: #{inspect(result)}")
    IO.puts("")

    # Test the NIF directly
    test_nif_integration()

    # Clean up
    send(handler_pid, :stop)
    IO.puts("✅ Example completed!")
  end

  # Test the NIF integration directly
  defp test_nif_integration do
    IO.puts("🔧 Testing NIF integration...")

    # Create test data
    test_data = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
    data_binary = Nx.to_binary(test_data)

    # Test the complete_infeed_request_fine NIF
    ref_id = "test_ref_123"

    case EXLA.NIF.complete_infeed_request_fine(ref_id, data_binary) do
      :ok ->
        IO.puts("  ✅ NIF call succeeded")

      {:error, reason} ->
        IO.puts("  ❌ NIF call failed: #{reason}")

      other ->
        IO.puts("  ❓ Unexpected NIF result: #{inspect(other)}")
    end
  end

  # Infeed handler that provides data when requested
  defp infeed_handler do
    receive do
      {:infeed_request, ref, shape} ->
        IO.puts("✅ Infeed handler received request!")
        IO.puts("  Reference: #{inspect(ref)}")
        IO.puts("  Shape: #{inspect(shape)}")

        # Generate response data
        response_tensor =
          case shape do
            [2, 2] ->
              Nx.tensor([[50.0, 60.0], [70.0, 80.0]], type: {:f, 32})

            _ ->
              Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})
          end

        IO.puts("  📤 Providing tensor: #{inspect(response_tensor)}")

        # Convert to binary and complete the request
        data_binary = Nx.to_binary(response_tensor)
        ref_binary = :erlang.term_to_binary(ref)

        # Complete the infeed request
        case EXLA.NIF.complete_infeed_request_fine(
               ref_binary,
               data_binary
             ) do
          :ok ->
            IO.puts("  ✅ Successfully completed infeed request")

          error ->
            IO.puts("  ❌ Failed to complete infeed: #{inspect(error)}")
        end

        infeed_handler()

      :stop ->
        IO.puts("🛑 Stopping infeed handler")

      other ->
        IO.puts("❓ Infeed handler received: #{inspect(other)}")
        infeed_handler()
    after
      5000 ->
        IO.puts("⏰ Infeed handler timeout")
        infeed_handler()
    end
  end
end

# Run the example
RealInfeedExample.run_example()
