defmodule Nx.Defn.MemorySplitter do
  @moduledoc """
  A module for splitting `Nx.Defn.Expr` based on memory constraints.

  This module uses `Nx.Defn.Graph.split/3` to split an expression graph
  into stages that fit within specified memory limits for each runner.
  """

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @doc """
  Splits an expression into stages based on memory constraints.

  ## Arguments

    * `expr` - The expression to split
    * `memory_limits` - A list of memory limits in bytes for each runner

  ## Returns

  A list of stages from `Nx.Defn.Graph.split/3` that respect the memory constraints.

  ## Examples

      memory_limits = [1_000_000, 2_000_000, 1_500_000]  # 1MB, 2MB, 1.5MB
      stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)
  """
  def split_by_memory(expr, memory_limits) when is_list(memory_limits) do
    initial_acc = %{
      memory_limits: memory_limits,
      current_runner: 0,
      current_memory: 0,
      # Track memory for each operation to avoid recalculation
      memory_cache: %{}
    }

    Nx.Defn.Graph.split(expr, &memory_split_fn/2, initial_acc)
  end

  defp memory_split_fn(%T{data: %Expr{id: id}} = tensor, acc) do
    %{
      memory_limits: memory_limits,
      current_runner: current_runner,
      current_memory: current_memory,
      memory_cache: memory_cache
    } = acc

    # Calculate memory required for this operation
    op_memory = calculate_operation_memory(tensor, memory_cache)

    # Cache the memory requirement
    memory_cache = Map.put(memory_cache, id, op_memory)

    # Check if adding this operation would exceed current runner's limit
    current_limit = Enum.at(memory_limits, current_runner, :infinity)

    cond do
      # If we're at the last runner or have infinite memory, don't split
      current_runner >= length(memory_limits) - 1 ->
        new_acc = %{acc | current_memory: current_memory + op_memory, memory_cache: memory_cache}
        {false, new_acc}

      # If adding this operation exceeds the limit, split and move to next runner
      current_memory + op_memory > current_limit ->
        new_acc = %{
          acc
          | current_runner: current_runner + 1,
            # Start fresh with this operation
            current_memory: op_memory,
            memory_cache: memory_cache
        }

        {true, new_acc}

      # Otherwise, accumulate memory and continue
      true ->
        new_acc = %{acc | current_memory: current_memory + op_memory, memory_cache: memory_cache}
        {false, new_acc}
    end
  end

  defp calculate_operation_memory(%T{} = tensor, memory_cache) do
    %T{data: %Expr{id: id, op: op, args: args}} = tensor

    # If already calculated, return cached value
    case Map.get(memory_cache, id) do
      nil ->
        # Calculate output memory
        output_memory = Nx.byte_size(tensor)

        # Calculate memory for intermediate values based on operation type
        intermediate_memory = estimate_intermediate_memory(op, args, tensor)

        # Total memory is output + intermediates
        output_memory + intermediate_memory

      cached_memory ->
        cached_memory
    end
  end

  defp estimate_intermediate_memory(op, args, output_tensor) do
    case op do
      # Some operations need temporary storage
      :dot ->
        # Matrix multiplication may need temporary storage for accumulation
        Nx.byte_size(output_tensor)

      :conv ->
        # Convolution needs temporary storage for the sliding window
        Nx.byte_size(output_tensor) * 2

      :fft ->
        # FFT typically needs complex temporary storage
        Nx.byte_size(output_tensor) * 2

      :sort ->
        # Sorting may need temporary storage
        Nx.byte_size(output_tensor)

      # Binary operations typically don't need extra memory beyond inputs
      op
      when op in [
             :add,
             :subtract,
             :multiply,
             :divide,
             :remainder,
             :atan2,
             :min,
             :max,
             :pow,
             :quotient,
             :bitwise_and,
             :bitwise_or,
             :bitwise_xor,
             :left_shift,
             :right_shift
           ] ->
        0

      # Unary operations typically operate in-place
      op
      when op in [
             :abs,
             :acos,
             :acosh,
             :asin,
             :asinh,
             :atan,
             :atanh,
             :ceil,
             :cos,
             :cosh,
             :exp,
             :floor,
             :log,
             :negate,
             :round,
             :sigmoid,
             :sign,
             :sin,
             :sinh,
             :sqrt,
             :tan,
             :tanh,
             :bitwise_not,
             :count_leading_zeros,
             :population_count
           ] ->
        0

      # Reductions might need temporary accumulators
      op when op in [:reduce, :reduce_max, :reduce_min, :reduce_sum, :reduce_product] ->
        # Estimate based on output size (usually much smaller than input)
        Nx.byte_size(output_tensor)

      # Default: assume no extra memory needed
      _ ->
        0
    end
  end

  @doc """
  Estimates the total memory usage of an expression.

  This can be useful for determining appropriate memory limits.
  """
  def estimate_memory_usage(expr) do
    acc = %{total_memory: 0, peak_memory: 0, memory_cache: %{}}

    # Walk through the expression tree
    {_result, final_acc} = traverse_for_memory(expr, acc)

    %{
      total_memory: final_acc.total_memory,
      peak_memory: final_acc.peak_memory
    }
  end

  defp traverse_for_memory(%T{data: %Expr{}} = tensor, acc) do
    memory = calculate_operation_memory(tensor, acc.memory_cache)

    new_acc = %{
      acc
      | total_memory: acc.total_memory + memory,
        peak_memory: max(acc.peak_memory, acc.total_memory + memory),
        memory_cache: Map.put(acc.memory_cache, tensor.data.id, memory)
    }

    {tensor, new_acc}
  end

  defp traverse_for_memory(other, acc), do: {other, acc}
end
