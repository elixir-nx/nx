defmodule Nx.Batch do
  @moduledoc """
  Creates a batch of tensors (and containers).

  A batch is lazily traversed, concatenated, and padded upon `defn` invocation.
  """

  @axis 0

  @doc """
  A Nx.Batch struct.

  The `:size` field is public.
  """
  @enforce_keys [:key]
  @derive {Inspect, only: [:size, :pad]}
  defstruct [:key, stack: [], size: 0, template: nil, pad: 0]

  @type t :: %Nx.Batch{
          stack: list(),
          size: non_neg_integer(),
          template: Nx.Container.t() | Nx.Tensor.t() | nil,
          pad: non_neg_integer(),
          key: term()
        }

  @doc """
  Returns a new empty batch.
  """
  def new, do: %Nx.Batch{key: :default}

  @doc """
  Sets the batch key for the given batch.
  """
  def key(%Nx.Batch{} = batch, key) do
    %{batch | key: key}
  end

  @doc """
  Merges two batches.

  The tensors on the left will appear before the tensors on the right.

  The size and padding of both batches are summed. The padding still
  applies only at the end of batch.

  It will raise if the batch templates are incompatible.

  ## Examples

      iex> batch1 = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)])
      iex> batch2 = Nx.Batch.concatenate([Nx.tensor([4, 5]), Nx.tensor([6, 7, 8])])
      iex> batch = Nx.Batch.merge(batch1, batch2)
      iex> batch.size
      8
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s32[8]
        [1, 2, 3, 4, 5, 6, 7, 8]
      >

  """
  def merge(left, right), do: merge([left, right])

  @doc """
  Merges a list of batches.

  See `merge/2`.
  """
  def merge([]), do: new()

  def merge([%Nx.Batch{} = head | tail]) do
    %{template: template, stack: stack, pad: pad, size: size, key: head_key} = head

    {template, stack, pad, size} =
      Enum.reduce(tail, {template, stack, pad, size}, fn batch, acc ->
        %Nx.Batch{template: template, stack: stack, pad: pad, size: size, key: tail_key} = batch
        {acc_template, acc_stack, acc_pad, acc_size} = acc

        if head_key != tail_key do
          raise ArgumentError,
                "cannot merge batches with different batch keys: #{inspect(head_key)} and #{inspect(tail_key)}"
        end

        if template != nil and acc_template != nil and not Nx.compatible?(template, acc_template) do
          raise ArgumentError, """
          cannot merge batches due to incompatible templates:

              #{inspect(template)}

          and:

              #{inspect(acc_template)}
          """
        end

        {acc_template || template, stack ++ acc_stack, pad + acc_pad, size + acc_size}
      end)

    %Nx.Batch{template: template, stack: stack, pad: pad, size: size, key: head_key}
  end

  @doc """
  Splits a batch in two, where the first one has at most `n` elements.

  If there is any padding and the batch is not full, the amount of padding
  necessary will be moved to the first batch and the remaining stays in the
  second batch.

  ## Examples

      iex> batch = Nx.Batch.concatenate([Nx.tensor([1, 2]), Nx.tensor([3, 4, 5])])
      iex> {left, right} = Nx.Defn.jit_apply(&Function.identity/1, [Nx.Batch.split(batch, 3)])
      iex> left
      #Nx.Tensor<
        s32[3]
        [1, 2, 3]
      >
      iex> right
      #Nx.Tensor<
        s32[2]
        [4, 5]
      >
  """
  def split(%Nx.Batch{} = batch, n) when is_integer(n) and n > 0 do
    %{template: template, stack: stack, pad: pad, size: size, key: key} = batch

    if n < size do
      {left, right} = drop_split(stack, size - n, [])

      {%{batch | stack: left, size: n, pad: 0},
       %Nx.Batch{template: template, pad: pad, size: size - n, stack: right, key: key}}
    else
      right_pad = max(size + pad - n, 0)
      left_pad = pad - right_pad
      {%{batch | pad: left_pad}, %Nx.Batch{template: template, pad: right_pad, key: key}}
    end
  end

  defp drop_split([{funs, size} | stack], n, acc) when size < n do
    drop_split(stack, n - size, [{funs, size} | acc])
  end

  defp drop_split([{funs, size} | stack], n, acc) when size == n do
    {stack, Enum.reverse([{funs, size} | acc])}
  end

  defp drop_split([{funs, size} | stack], n, acc) when size > n do
    left_start = 0
    left_size = size - n

    left_funs =
      Enum.map(funs, fn fun ->
        fn -> Nx.slice_along_axis(fun.(), left_start, left_size, axis: @axis) end
      end)

    right_start = size - n
    right_size = n

    right_funs =
      Enum.map(funs, fn fun ->
        fn -> Nx.slice_along_axis(fun.(), right_start, right_size, axis: @axis) end
      end)

    {[{left_funs, left_size} | stack], Enum.reverse([{right_funs, right_size} | acc])}
  end

  @doc """
  Configures the batch with the given padding.

  The batch will be padded when consumed:

      iex> batch = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [Nx.Batch.pad(batch, 2)])
      #Nx.Tensor<
        s32[5]
        [1, 2, 3, 0, 0]
      >
  """
  def pad(%Nx.Batch{} = batch, pad) when is_integer(pad) and pad >= 0 do
    %{batch | pad: pad}
  end

  @doc """
  Concatenates the given entries to the batch.

  Entries are concatenated based on their first axis.
  If the first axis has multiple entries, each entry
  is added to the size of the batch.

  You can either concatenate to an existing batch
  or skip the batch argument to create a new batch.

  See `stack/2` if you want to stack entries instead
  of concatenating them.

  ## Examples

  If no batch is given, one is automatically created:

      iex> batch = Nx.Batch.concatenate([Nx.tensor([1]), Nx.tensor([2]), Nx.tensor([3])])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s32[3]
        [1, 2, 3]
      >

  But you can also concatenate to existing batches:

      iex> batch = Nx.Batch.concatenate([Nx.tensor([1]), Nx.tensor([2])])
      iex> batch = Nx.Batch.concatenate(batch, [Nx.tensor([3]), Nx.tensor([4])])
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s32[4]
        [1, 2, 3, 4]
      >

  If the first axis has multiple entries, each entry counts
  towards the size of the batch:

      iex> batch = Nx.Batch.concatenate([Nx.tensor([1, 2]), Nx.tensor([3, 4, 5])])
      iex> batch.size
      5
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s32[5]
        [1, 2, 3, 4, 5]
      >

  What makes batches powerful is that they can concatenate
  across containers:

      iex> container1 = {Nx.tensor([11]), Nx.tensor([21])}
      iex> container2 = {Nx.tensor([12]), Nx.tensor([22])}
      iex> batch = Nx.Batch.concatenate([container1, container2])
      iex> {batched1, batched2} = Nx.Defn.jit_apply(&Function.identity/1, [batch])
      iex> batched1
      #Nx.Tensor<
        s32[2]
        [11, 12]
      >
      iex> batched2
      #Nx.Tensor<
        s32[2]
        [21, 22]
      >

  """
  def concatenate(%Nx.Batch{} = batch \\ new(), entries) when is_list(entries),
    do: add(batch, entries, false)

  @doc """
  Stacks the given entries to the batch.

  Each entry counts exactly as a single entry.
  You can either stack to an existing batch
  or skip the batch argument to create a new batch.

  See `concatenate/2` if you want to concatenate entries
  instead of stacking them.

  ## Examples

  If no batch is given, one is automatically created:

      iex> batch = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)])
      iex> batch.size
      3
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s32[3]
        [1, 2, 3]
      >

  But you can also stack an existing batch:

      iex> batch = Nx.Batch.stack([Nx.tensor(1), Nx.tensor(2)])
      iex> batch = Nx.Batch.stack(batch, [Nx.tensor(3), Nx.tensor(4)])
      iex> batch.size
      4
      iex> Nx.Defn.jit_apply(&Function.identity/1, [batch])
      #Nx.Tensor<
        s32[4]
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
        s32[2]
        [11, 12]
      >
      iex> batched2
      #Nx.Tensor<
        s32[2]
        [21, 22]
      >

  """
  def stack(%Nx.Batch{} = batch \\ new(), entries) when is_list(entries),
    do: add(batch, entries, true)

  defp add(batch, [], _new_axis?), do: batch

  defp add(batch, [head | tail], new_axis?) do
    %{template: template, stack: stack, size: size} = batch
    {head_template, head_size, head_funs} = traverse(head, new_axis?)
    acc = {head_size + size, [{head_funs, head_size} | stack]}

    {size, stack} =
      Enum.reduce(tail, acc, fn arg, {acc_size, acc_stack} ->
        {arg_template, size, arg_funs} = traverse(arg, new_axis?)

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

        {size + acc_size, [{arg_funs, size} | acc_stack]}
      end)

    if template == nil or Nx.compatible?(template, head_template) do
      %{batch | template: head_template, stack: stack, size: size}
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

  defp traverse(container, true) do
    {template, funs} =
      Nx.LazyContainer.traverse(container, [], fn template, fun, acc ->
        {template, [fn -> Nx.new_axis(fun.(), @axis) end | acc]}
      end)

    {template, 1, funs}
  end

  defp traverse(container, false) do
    {template, {size, funs}} =
      Nx.LazyContainer.traverse(container, {nil, []}, fn template, fun, {acc_size, acc_funs} ->
        %Nx.Tensor{shape: shape, names: names} = template

        if shape == {} do
          raise ArgumentError, "cannot concatenate scalar tensor in #{inspect(container)}"
        end

        size = elem(shape, @axis)

        if acc_size != nil and size != acc_size do
          raise ArgumentError,
                "concatenate expects all tensors in the same container to have the same value " <>
                  "for first axis, got #{size} and #{acc_size} in #{inspect(container)}"
        end

        template = %{template | shape: Tuple.delete_at(shape, @axis), names: tl(names)}
        {template, {size, [fun | acc_funs]}}
      end)

    if size == nil do
      raise ArgumentError, "cannot have an empty container in concatenate: #{inspect(container)}"
    end

    {template, size, funs}
  end
end

defimpl Nx.LazyContainer, for: Nx.Batch do
  @axis 0

  def traverse(%{stack: []}, _acc, _acc_fun) do
    raise ArgumentError, "cannot traverse/jit/compile Nx.Batch without entries"
  end

  def traverse(%{stack: funs_size, pad: pad, template: template, size: size}, acc, acc_fun) do
    total = size + pad

    funs =
      funs_size
      |> first_reverse([])
      |> Enum.zip_with(fn funs ->
        fn ->
          funs
          |> apply_each()
          |> Nx.concatenate(axis: @axis)
          |> maybe_pad(pad)
        end
      end)
      |> Enum.reverse()

    {template, {acc, []}} =
      Nx.Defn.Composite.traverse(template, {acc, funs}, fn template, {acc, [fun | funs]} ->
        %{shape: shape, names: names} = template
        template = %{template | shape: Tuple.insert_at(shape, 0, total), names: [nil | names]}
        {template, acc} = acc_fun.(template, fun, acc)
        {template, {acc, funs}}
      end)

    {template, acc}
  end

  defp first_reverse([{fun, _} | funs], acc), do: first_reverse(funs, [fun | acc])
  defp first_reverse([], acc), do: acc

  defp apply_each([fun | funs]), do: [fun.() | apply_each(funs)]
  defp apply_each([]), do: []

  defp maybe_pad(tensor, 0), do: tensor

  defp maybe_pad(tensor, pad_size) do
    padding =
      {0, 0, 0}
      |> List.duplicate(Nx.rank(tensor))
      |> List.replace_at(@axis, {0, pad_size, 0})

    Nx.pad(tensor, Nx.tensor(0, type: Nx.type(tensor), backend: Nx.BinaryBackend), padding)
  end
end
