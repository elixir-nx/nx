# Message-Based Communication System

This document describes the new message-based communication system that replaces traditional StableHLO `infeed` and `outfeed` operations with direct message passing to Elixir processes.

## Overview

The traditional EXLA system uses StableHLO's `infeed` and `outfeed` operations along with tokens to communicate between the compiled computation and Elixir processes. This new system replaces those operations with custom calls that send messages directly to Elixir processes using the Erlang NIF message passing capabilities.

## Key Benefits

1. **Direct Communication**: Messages are sent directly to Elixir processes without intermediate queues
2. **Simplified Protocol**: No need for complex token management and flag-based protocols
3. **Better Integration**: Leverages Erlang's native message passing for more natural integration
4. **Arbitrary Tensor Lists**: Supports sending/receiving multiple tensors in a single operation

## Architecture

### Custom Calls

The system implements several custom calls:

- `message_infeed_f32`, `message_infeed_f64`, `message_infeed_s32`: Receive data from Elixir processes
- `message_outfeed_f32`, `message_outfeed_f64`, `message_outfeed_s32`: Send data to Elixir processes

Each custom call takes:

1. A PID buffer (u8 tensor containing the serialized Elixir process PID)
2. Data buffer(s) (for outfeed) or returns data buffer (for infeed)
3. Returns a confirmation buffer

### Elixir Integration

#### EXLA.Defn.MessageComm Module

This module provides the Elixir-side interface:

```elixir
# Create message communication context
message_comm = EXLA.Defn.MessageComm.new(function)

# Send tensor data
MessageComm.send_tensor(message_comm, tensor_value)

# Send multiple tensors
MessageComm.send_tensors(message_comm, [tensor1, tensor2, tensor3])

# Receive tensor data
tensor = MessageComm.receive_tensor(message_comm, typespec)
```

#### Value Module Functions

New functions in `EXLA.MLIR.Value`:

```elixir
# Message-based infeed
Value.message_infeed(pid_value, typespec)

# Message-based outfeed (single tensor)
Value.message_outfeed(pid_value, data_value)

# Message-based outfeed (multiple tensors)
Value.message_multi_outfeed(pid_value, data_values)
```

## Usage

### Enabling Message-Based Communication

Enable message-based communication by setting the `:use_message_comm` option:

```elixir
# In defn compilation options
Nx.Defn.compile(my_function, use_message_comm: true)

# Or in client configuration
Application.put_env(:exla, :clients,
  host: [platform: :host, use_message_comm: true]
)
```

### Example: Simple Outfeed

```elixir
defmodule Example do
  import Nx.Defn

  defn compute_and_send(tensor) do
    result = Nx.add(tensor, 1)
    # Use metadata to trigger message-based outfeed
    Nx.metadata(result, %{exla_custom_call: :process_pid})
  end

  def run do
    input = Nx.tensor([1.0, 2.0, 3.0])

    # Start listening for messages
    Task.async(fn ->
      receive do
        {:outfeed_data, binary} ->
          IO.puts("Received: #{inspect(binary)}")
      end
    end)

    # Run computation with message communication enabled
    result = compute_and_send(input)
    IO.puts("Computation result: #{inspect(result)}")
  end
end
```

### Example: Multiple Tensor Outfeed

```elixir
defn compute_multiple(tensor) do
  result1 = Nx.add(tensor, 1)
  result2 = Nx.multiply(tensor, 2)

  # Send multiple tensors via message passing
  # This would be implemented using the multi-outfeed custom call
  {result1, result2}
end
```

## Message Protocol

### Outfeed Messages

Single tensor outfeed:

```elixir
{:outfeed_data, binary}
```

Multiple tensor outfeed:

```elixir
{:outfeed_multi_data, [binary1, binary2, binary3]}
```

### Infeed Messages

Infeed request (sent from custom call to Elixir):

```elixir
{:infeed_request, reference}
```

Infeed response (sent from Elixir to custom call):

```elixir
{:infeed_response, reference, binary_data}
```

## Implementation Details

### Custom Call Registration

Each data type requires specific custom call handlers:

```cpp
// F32 handlers
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_outfeed_f32", "Host", message_outfeed_f32);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_infeed_f32", "Host", message_infeed_f32);

// F64 handlers
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_outfeed_f64", "Host", message_outfeed_f64);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_infeed_f64", "Host", message_infeed_f64);

// S32 handlers
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_outfeed_s32", "Host", message_outfeed_s32);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_infeed_s32", "Host", message_infeed_s32);
```

### PID Encoding

Process PIDs are encoded as u8 tensors using Erlang's term format:

```elixir
pid_binary = :erlang.term_to_binary(self())
pid_data = :erlang.binary_to_list(pid_binary)
pid_typespec = Typespec.tensor({:u, 8}, {length(pid_data)})
pid_value = Value.constant(function, pid_data, pid_typespec)
```

### Tensor Data Encoding

Tensor data is sent as binary messages containing the raw tensor data:

```cpp
// Create binary from tensor data
ErlNifBinary binary;
enif_alloc_binary(data_size * sizeof(DataType), &binary);
const DataType *data = data_buffer.typed_data();
std::memcpy(binary.data, data, data_size * sizeof(DataType));
```

## Limitations and Future Work

### Current Limitations

1. **Synchronous Operation**: The current implementation doesn't wait for infeed responses
2. **Limited Types**: Only f32, f64, and s32 are currently implemented
3. **Host Platform Only**: Currently only supports host platform (CPU)

### Future Enhancements

1. **Asynchronous Infeed**: Implement proper message queuing for infeed operations
2. **More Data Types**: Add support for f16, bf16, u32, u64, etc.
3. **GPU Support**: Extend to CUDA and ROCm platforms
4. **Complex Tensors**: Support for complex number tensors
5. **Streaming**: Support for streaming large tensors in chunks

## Migration Guide

### From Traditional Outfeed

**Before:**

```elixir
defn my_function(tensor) do
  result = Nx.add(tensor, 1)
  Nx.Defn.print(result, label: "debug")
  result
end
```

**After:**

```elixir
defn my_function(tensor) do
  result = Nx.add(tensor, 1)
  Nx.metadata(result, %{exla_custom_call: :process_pid})
end

# Listen for messages in your process
receive do
  {:outfeed_data, binary} -> handle_tensor_data(binary)
end
```

### From Traditional Infeed

**Before:**

```elixir
# Complex infeed setup with tokens and hooks
```

**After:**

```elixir
# Simplified message-based approach
defn my_function() do
  # Receive data via message passing
  # Implementation would use message_infeed custom call
end
```

## Testing

To test the message-based communication system:

1. Enable message communication in your client configuration
2. Use the provided example functions
3. Monitor your process mailbox for incoming messages
4. Verify that tensor data is correctly transmitted

## Performance Considerations

- Message passing adds some overhead compared to direct memory operations
- For large tensors, consider chunking or streaming approaches
- The system is optimized for host platform usage
- GPU implementations may require additional synchronization
