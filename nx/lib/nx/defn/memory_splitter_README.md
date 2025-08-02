# Memory-Based Expression Splitting in Nx

This module provides functionality to split Nx computational graphs based on memory constraints, enabling efficient distribution of computations across devices with different memory capacities.

## Overview

The `Nx.Defn.MemorySplitter` module builds on top of `Nx.Defn.Graph` to provide intelligent splitting of expression graphs based on memory requirements. This is particularly useful for:

- Distributing large models across multiple GPUs with different memory sizes
- Running computations on memory-constrained devices
- Optimizing memory usage in distributed computing scenarios

## Key Features

1. **Automatic Memory Estimation**: Uses `Nx.byte_size/1` to calculate memory requirements for each operation
2. **Intelligent Splitting**: Respects memory limits while minimizing the number of splits
3. **Accumulator Support**: The underlying `Nx.Defn.Graph.split/3` now supports stateful splitting functions
4. **Operation-Aware**: Accounts for temporary memory needed by operations like matrix multiplication and convolution

## Usage

### Basic Example

```elixir
# Define your computation
expr = Nx.Defn.debug_expr(fn x, y ->
  x
  |> Nx.dot(y)
  |> Nx.add(1)
  |> Nx.sigmoid()
end).(input_tensor, weight_tensor)

# Define memory limits for each runner (in bytes)
memory_limits = [
  1_000_000,   # Runner 1: 1MB
  2_000_000,   # Runner 2: 2MB
  1_500_000    # Runner 3: 1.5MB
]

# Split the expression
stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)

# Execute the stages
result = Nx.Defn.Graph.run(stages, [input_tensor, weight_tensor])
```

### Memory Estimation

Before splitting, you can estimate the memory requirements:

```elixir
%{total_memory: total, peak_memory: peak} =
  Nx.Defn.MemorySplitter.estimate_memory_usage(expr)

IO.puts("Total memory needed: #{total} bytes")
IO.puts("Peak memory needed: #{peak} bytes")
```

### Advanced: Custom Split Functions with Accumulator

The enhanced `Nx.Defn.Graph.split/3` function now supports accumulator-based splitting:

```elixir
# Custom split function that tracks operation count
split_fn = fn tensor, {count, max_ops} ->
  new_count = count + 1
  should_split = new_count >= max_ops
  new_acc = if should_split, do: {0, max_ops}, else: {new_count, max_ops}
  {should_split, new_acc}
end

# Split every 5 operations
stages = Nx.Defn.Graph.split(expr, split_fn, {0, 5})
```

## Implementation Details

### Memory Calculation

The memory splitter calculates memory requirements by:

1. **Output Memory**: Direct calculation using `Nx.byte_size/1`
2. **Intermediate Memory**: Operation-specific estimates for temporary buffers
   - Matrix operations (`:dot`): 1x output size
   - Convolutions (`:conv`): 2x output size
   - FFT operations (`:fft`): 2x output size
   - Reductions: Size of accumulator (output size)
   - Element-wise operations: No additional memory

### Splitting Algorithm

The algorithm maintains state across the traversal:

```elixir
%{
  memory_limits: [limit1, limit2, ...],
  current_runner: 0,
  current_memory: 0,
  memory_cache: %{}
}
```

When an operation would exceed the current runner's limit, it triggers a split and moves to the next runner.

## Best Practices

1. **Buffer Space**: Add 10-20% buffer to memory limits to account for framework overhead
2. **Profile First**: Use `estimate_memory_usage/1` to understand memory requirements
3. **Balance Runners**: Try to distribute memory limits evenly when possible
4. **Test Splits**: Verify that split computations produce the same results as non-split versions

## Limitations

- Memory estimates are approximations and may not account for all backend-specific allocations
- The splitter assumes sequential execution of stages
- Splitting may introduce communication overhead between stages

## Future Enhancements

- Support for parallel stage execution
- More sophisticated memory models for different backends
- Automatic memory limit discovery
- Cost-based optimization (considering both memory and computation)
