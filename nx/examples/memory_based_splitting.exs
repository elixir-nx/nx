# Example of memory-based expression splitting in Nx
#
# This example shows how to split a computational graph based on memory
# constraints for different runners/devices.

# First, let's create a complex expression that simulates a neural network layer
defmodule MemorySplittingExample do
  import Nx.Defn

  # A simple neural network layer computation
  defn layer(input, weight, bias) do
    input
    |> Nx.dot(weight)
    |> Nx.add(bias)
    |> Nx.sigmoid()
  end

  # A more complex computation with multiple layers
  defn multi_layer_network(input, w1, b1, w2, b2, w3, b3) do
    input
    |> layer(w1, b1)
    |> layer(w2, b2)
    |> layer(w3, b3)
    |> Nx.sum()
  end

  def run_example() do
    # Create sample tensors
    batch_size = 32
    input_size = 784  # e.g., flattened 28x28 image
    hidden1_size = 256
    hidden2_size = 128
    output_size = 10

    # Input and weights
    input = Nx.random_normal({batch_size, input_size}, type: :f32)
    w1 = Nx.random_normal({input_size, hidden1_size}, type: :f32) |> Nx.multiply(0.01)
    b1 = Nx.zeros({hidden1_size}, type: :f32)
    w2 = Nx.random_normal({hidden1_size, hidden2_size}, type: :f32) |> Nx.multiply(0.01)
    b2 = Nx.zeros({hidden2_size}, type: :f32)
    w3 = Nx.random_normal({hidden2_size, output_size}, type: :f32) |> Nx.multiply(0.01)
    b3 = Nx.zeros({output_size}, type: :f32)

    # Get the expression without evaluating it
    expr = Nx.Defn.debug_expr(&multi_layer_network/7).(input, w1, b1, w2, b2, w3, b3)

    # Estimate memory usage
    IO.puts("Estimating memory usage...")
    %{total_memory: total, peak_memory: peak} = Nx.Defn.MemorySplitter.estimate_memory_usage(expr)
    IO.puts("Total memory needed: #{format_bytes(total)}")
    IO.puts("Peak memory needed: #{format_bytes(peak)}")
    IO.puts("")

    # Example 1: Split across 3 devices with different memory capacities
    IO.puts("Example 1: Splitting across devices with different memory limits")
    memory_limits = [
      500_000,    # Device 1: 500KB
      1_000_000,  # Device 2: 1MB
      2_000_000   # Device 3: 2MB
    ]

    stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)
    IO.puts("Number of stages created: #{length(stages)}")

    Enum.with_index(stages, fn stage, idx ->
      IO.puts("  Stage #{idx + 1}: #{inspect(stage.expr.data.op)} operation")
    end)
    IO.puts("")

    # Example 2: Equal memory distribution
    IO.puts("Example 2: Equal memory distribution across 4 runners")
    num_runners = 4
    memory_per_runner = div(peak, num_runners) + 100_000  # Add some buffer
    equal_limits = List.duplicate(memory_per_runner, num_runners)

    stages2 = Nx.Defn.MemorySplitter.split_by_memory(expr, equal_limits)
    IO.puts("Number of stages with equal distribution: #{length(stages2)}")
    IO.puts("")

    # Example 3: Running the split computation
    IO.puts("Example 3: Executing the split computation")
    args = [input, w1, b1, w2, b2, w3, b3]

    # Original computation
    IO.puts("Computing original (non-split) result...")
    original_result = Nx.Defn.jit_apply(&multi_layer_network/7, args)

    # Split computation
    IO.puts("Computing split result...")
    split_result = Nx.Defn.Graph.run(stages, args)

    # Verify they match
    diff = Nx.subtract(original_result, split_result) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
    IO.puts("Difference between original and split computation: #{diff}")

    if diff < 1.0e-5 do
      IO.puts("✓ Results match!")
    else
      IO.puts("✗ Results differ significantly!")
    end
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 2)} KB"
  defp format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024), 2)} MB"
end

# Run the example
MemorySplittingExample.run_example()
