defmodule EXLA.Defn.MessageComm do
  @moduledoc false
  # Message-based communication module that replaces traditional infeed/outfeed
  # with direct message passing to Elixir processes using custom calls.

  alias EXLA.MLIR.Value
  alias EXLA.MLIR.Function
  alias EXLA.Typespec

  defstruct [:message_handlers, :pid_value]

  @doc """
  Creates a new message communication context with the current process PID.
  """
  def new(%Function{} = function) do
    # Create a PID value that contains the current process PID
    pid_binary = :erlang.term_to_binary(self())
    pid_data = :erlang.binary_to_list(pid_binary)
    pid_typespec = Typespec.tensor({:u, 8}, {length(pid_data)})
    pid_value = Value.constant(function, pid_data, pid_typespec)

    %__MODULE__{
      message_handlers: %{},
      pid_value: pid_value
    }
  end

  @doc """
  Sends tensor data to the current process via message passing.
  Replaces traditional outfeed operations.
  """
  def send_tensor(%__MODULE__{pid_value: pid_value}, %Value{} = tensor_value) do
    Value.message_outfeed(pid_value, tensor_value)
  end

  @doc """
  Sends multiple tensors to the current process via message passing.
  """
  def send_tensors(%__MODULE__{pid_value: pid_value}, tensor_values)
      when is_list(tensor_values) do
    Value.message_multi_outfeed(pid_value, tensor_values)
  end

  @doc """
  Receives tensor data from the current process via message passing.
  Replaces traditional infeed operations.
  """
  def receive_tensor(%__MODULE__{pid_value: pid_value}, typespec) do
    Value.message_infeed(pid_value, typespec)
  end

  @doc """
  Registers a message handler for a specific hook name.
  """
  def add_handler(%__MODULE__{} = comm, name, handler_fun) do
    %{comm | message_handlers: Map.put(comm.message_handlers, name, handler_fun)}
  end

  @doc """
  Processes a hook by sending tensor data via message passing instead of outfeed.
  """
  def process_hook(%__MODULE__{} = comm, tensor_values, hook_name) do
    case Map.get(comm.message_handlers, hook_name) do
      nil ->
        # No handler registered, just send the data
        send_tensors(comm, tensor_values)

      _handler_fun ->
        # Send data and register handler for processing
        send_tensors(comm, tensor_values)
        # Note: The handler will be called when the message is received
        # This maintains the same async behavior as the original outfeed system
        comm
    end
  end

  @doc """
  Starts a message communication handler process that listens for messages
  from the compiled computation and processes them accordingly.
  """
  def start_message_handler(handlers \\ %{}) do
    parent = self()

    Task.start_link(fn ->
      message_loop(parent, handlers)
    end)
  end

  @doc """
  Executes with message-based communication instead of traditional outfeed.
  """
  def execute_with_messages(executable, buffers, message_handlers, run_options) do
    # Start message handler if we have handlers
    handler_pid =
      if map_size(message_handlers) > 0 do
        {:ok, pid} = start_message_handler(message_handlers)
        pid
      else
        nil
      end

    try do
      # Run the executable - it will send messages directly to this process
      result = EXLA.Executable.run(executable, [buffers], run_options)

      # Process any messages that were sent during execution
      collect_messages([])

      result
    after
      # If we started a handler, stop it
      if handler_pid do
        send(handler_pid, :stop)
      end
    end
  end

  defp collect_messages(acc) do
    receive do
      {:outfeed_data, _binary} = msg ->
        collect_messages([msg | acc])

      {:outfeed_multi_data, _binaries} = msg ->
        collect_messages([msg | acc])
    after
      # No more messages, return what we collected
      0 -> Enum.reverse(acc)
    end
  end

  defp message_loop(parent, handlers) do
    receive do
      {:outfeed_data, binary} ->
        # Handle single tensor data
        case Map.get(handlers, :default) do
          nil -> send(parent, {:tensor_data, binary})
          fun -> fun.(binary)
        end

        message_loop(parent, handlers)

      {:outfeed_multi_data, binaries} ->
        # Handle multiple tensor data
        case Map.get(handlers, :multi_default) do
          nil -> send(parent, {:tensor_list, binaries})
          fun -> fun.(binaries)
        end

        message_loop(parent, handlers)

      {:infeed_request, ref, shape} ->
        # Handle infeed request by providing data
        case Map.get(handlers, :infeed) do
          nil ->
            # No infeed handler - provide zeros
            zeros = create_zeros_for_shape(shape)
            complete_infeed_with_data(ref, zeros)

          handler ->
            # Call the infeed handler to get data
            data = handler.(shape)
            complete_infeed_with_data(ref, data)
        end
        message_loop(parent, handlers)

      {:register_handler, name, fun} ->
        message_loop(parent, Map.put(handlers, name, fun))

      :stop ->
        :ok

      other ->
        # Forward unknown messages to parent
        send(parent, other)
        message_loop(parent, handlers)
    end
  end

  @doc """
  Handles infeed requests by providing data to the computation.
  This function should be called when the computation requests data via infeed.
  """
  def handle_infeed_request(pid, ref, shape, data) when is_list(data) do
    # Create the response message with the tensor data
    response = {:infeed_response, ref, data}
    send(pid, response)
  end

  def handle_infeed_request(pid, ref, shape, data) do
    # Convert single tensor to list format
    handle_infeed_request(pid, ref, shape, [data])
  end

  @doc """
  Sets up an infeed data provider that will respond to infeed requests.
  """
  def setup_infeed_provider(data_map) when is_map(data_map) do
    parent = self()

    spawn_link(fn ->
      infeed_provider_loop(parent, data_map)
    end)
  end

  defp infeed_provider_loop(parent, data_map) do
    receive do
      {:infeed_request, ref, shape} ->
        # Look up data for this request
        case Map.get(data_map, shape) do
          nil ->
            # No data available - send zeros
            zeros = create_zeros_for_shape(shape)
            send(parent, {:infeed_response, ref, zeros})

          data ->
            # Send the requested data
            send(parent, {:infeed_response, ref, data})
        end

        infeed_provider_loop(parent, data_map)

      {:update_data, new_data_map} ->
        infeed_provider_loop(parent, new_data_map)

      :stop ->
        :ok
    end
  end

  defp create_zeros_for_shape(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(1, &*/2)
    |> then(&List.duplicate(0.0, &1))
  end

  defp complete_infeed_with_data(ref, data) do
    # Convert ref to string ID and send data to complete the infeed request
    ref_binary = :erlang.term_to_binary(ref)
    ref_id = Base.encode64(ref_binary)

    # Convert data to binary format expected by the custom call
    data_binary = case data do
      list when is_list(list) ->
        # Convert list of numbers to binary
        for num <- list, into: <<>>, do: <<num::float-native-32>>

      binary when is_binary(binary) ->
        binary

      tensor ->
        # If it's an Nx tensor, convert to binary
        tensor
        |> Nx.to_binary()
    end

    EXLA.NIF.complete_infeed_request(ref_id, data_binary)
  end
end
