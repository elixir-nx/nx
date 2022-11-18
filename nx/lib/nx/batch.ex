defmodule Nx.Batch do
  @moduledoc """
  Creates a batch of tensors (and containers).

  A batch is lazily traversed, concatenated, and padded upon `defn` invocation.
  """

  @derive {Inspect, only: [:size, :pad]}
  defstruct stack: [], size: 0, template: nil, pad: 0

  @doc """
  Returns a new empty batch.
  """
  def new, do: %Nx.Batch{}

  @doc """
  Configures the batch with the given padding.

  The batch will be padded when consumed:

      iex> batch = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [Nx.Batch.pad(batch, 2)])
      #Nx.Tensor<
        s64[5]
        [1, 2, 3, 0, 0]
      >
  """
  def pad(%Nx.Batch{} = batch, pad) when is_integer(pad) and pad >= 0 do
    %{batch | pad: pad}
  end

  @doc """
  Concatenates the given entries to the batch.

  You can either concatenate to an existing batch
  or skip the batch argument to create a new batch.

  See `stack/2` if you want to stack entries instead
  of concatenating them.

  ## Examples

  If no batch is given, one is automatically created:

      iex> batch = Nx.Batch.concatenate([Nx.tensor([1]), Nx.tensor([2]), Nx.tensor([3])])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

  But you can also concatenate to existing batches:

      iex> batch = Nx.Batch.concatenate([Nx.tensor([1]), Nx.tensor([2])])
      iex> batch = Nx.Batch.concatenate(batch, [Nx.tensor([3]), Nx.tensor([4])])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s64[4]
        [1, 2, 3, 4]
      >

  What makes batches powerful is that they can concatenate
  across containers:

      iex> container1 = {Nx.tensor([11]), Nx.tensor([21])}
      iex> container2 = {Nx.tensor([12]), Nx.tensor([22])}
      iex> batch = Nx.Batch.concatenate([container1, container2])
      iex> {batched1, batched2} = Nx.Defn.jit_apply(&Function.identity/1, [batch])
      iex> batched1
      #Nx.Tensor<
        s64[2]
        [11, 12]
      >
      iex> batched2
      #Nx.Tensor<
        s64[2]
        [21, 22]
      >

  """
  def concatenate(%Nx.Batch{} = batch \\ new(), entries), do: add(batch, entries, false)

  @doc """
  Stacks the given entries to the batch.

  You can either stack to an existing batch
  or skip the batch argument to create a new batch.

  See `concatenate/2` if you want to concatenate entries
  instead of stacking them.

  ## Examples

  If no batch is given, one is automatically created:

      iex> batch = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

  But you can also stack an existing batch:

      iex> batch = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2)])
      iex> batch = Nx.Batch.stack(batch, [Nx.tensor(3), Nx.tensor(4)])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s64[4]
        [1, 2, 3, 4]
      >

  What makes batches powerful is that they can concatenate
  across containers:

      iex> container1 = {Nx.tensor(11), Nx.tensor(21)}
      iex> container2 = {Nx.tensor(12), Nx.tensor(22)}
      iex> batch = Nx.Batch.stack([container1, container2])
      iex> {batched1, batched2} = Nx.Defn.jit_apply(&Function.identity/1, [batch])
      iex> batched1
      #Nx.Tensor<
        s64[2]
        [11, 12]
      >
      iex> batched2
      #Nx.Tensor<
        s64[2]
        [21, 22]
      >

  """
  def stack(%Nx.Batch{} = batch \\ new(), entries), do: add(batch, entries, true)

  defp add(batch, [], _new_axis?), do: batch

  defp add(batch, [head | tail], new_axis?) do
    %{template: template, stack: stack, size: size} = batch
    {head_template, head_funs} = traverse(head, new_axis?)

    stack =
      Enum.reduce(tail, [head_funs | stack], fn arg, acc ->
        {arg_template, arg_funs} = traverse(arg, new_axis?)

        unless Nx.compatible?(arg_template, head_template) do
          raise ArgumentError, """
          cannot add to batch due to incompatible tensors/containers.

          The head of the list has shape:

          #{inspect(head_template)}

          But another list element has template:

          #{inspect(arg_template)}

          From entry:

          #{inspect(arg)}
          """
        end

        [arg_funs | acc]
      end)

    if template == nil or Nx.compatible?(template, head_template) do
      %{batch | template: head_template, stack: stack, size: length(tail) + 1 + size}
    else
      raise ArgumentError, """
      cannot add to batch due to incompatible tensors/containers.

      The batch has shape:

      #{inspect(template)}

      But then the head of the list has template:

      #{inspect(head_template)}

      From entry:

      #{inspect(head)}
      """
    end
  end

  defp traverse(container, new_axis?) do
    Nx.LazyContainer.traverse(container, [], &{&1, [{&2, new_axis?} | &3]})
  end
end

defimpl Nx.LazyContainer, for: Nx.Batch do
  def traverse(%{stack: funs_new_axis, pad: pad, template: template}, acc, acc_fun) do
    funs =
      funs_new_axis
      |> Enum.zip_with(fn funs ->
        fn ->
          funs
          |> apply_fun_new_axis([])
          |> Nx.concatenate(axis: 0)
          |> maybe_pad(pad)
        end
      end)
      |> Enum.reverse()

    {template, {acc, []}} =
      Nx.Defn.Composite.traverse(template, {acc, funs}, fn template, {acc, [fun | funs]} ->
        {template, acc} = acc_fun.(template, fun, acc)
        {template, {acc, funs}}
      end)

    {template, acc}
  end

  defp apply_fun_new_axis([{fun, true} | funs], acc),
    do: apply_fun_new_axis(funs, [Nx.new_axis(fun.(), 0) | acc])

  defp apply_fun_new_axis([{fun, false} | funs], acc),
    do: apply_fun_new_axis(funs, [fun.() | acc])

  defp apply_fun_new_axis([], acc), do: acc

  defp maybe_pad(tensor, 0), do: tensor

  defp maybe_pad(tensor, pad_size) do
    first_axis_pad = {0, pad_size, 0}
    rest_axes_pad = List.duplicate({0, 0, 0}, Nx.rank(tensor) - 1)
    Nx.pad(tensor, Nx.tensor(0, type: Nx.type(tensor)), [first_axis_pad | rest_axes_pad])
  end
end
