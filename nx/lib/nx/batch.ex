defmodule Nx.Batch do
  @moduledoc """
  A data-structure returned by `Nx.Defn.batch/2`.
  """
  @enforce_keys [:list, :pad]
  defstruct [:list, :pad]
end

defimpl Nx.LazyContainer, for: Nx.Batch do
  def traverse(%{list: [head | tail], pad: pad}, acc, acc_fun) do
    {head_template, head_funs} = Nx.LazyContainer.traverse(head, [], &{&1, [&2 | &3]})

    tail_funs =
      Enum.map(tail, fn arg ->
        {arg_template, arg_funs} = Nx.LazyContainer.traverse(arg, [], &{&1, [&2 | &3]})

        unless Nx.compatible?(arg_template, head_template) do
          raise ArgumentError, """
          batch is made of incompatible tensors/containers.

          The head of the batch has shape:

          #{inspect(head_template)}

          But then got template:

          #{inspect(arg_template)}

          From entry:

          #{inspect(arg)}
          """
        end

        arg_funs
      end)

    funs =
      [head_funs | tail_funs]
      |> Enum.zip_with(fn funs ->
        fn ->
          funs
          |> apply_fun_new_axis()
          |> Nx.concatenate(axis: 0)
          |> maybe_pad(pad)
        end
      end)
      |> Enum.reverse()

    {template, {acc, []}} =
      Nx.Defn.Composite.traverse(head_template, {acc, funs}, fn template, {acc, [fun | funs]} ->
        {template, acc} = acc_fun.(template, fun, acc)
        {template, {acc, funs}}
      end)

    {template, acc}
  end

  defp apply_fun_new_axis([fun | funs]), do: [fun.() |> Nx.new_axis(0) | apply_fun_new_axis(funs)]
  defp apply_fun_new_axis([]), do: []

  defp maybe_pad(tensor, 0), do: tensor

  defp maybe_pad(tensor, pad_size) do
    first_axis_pad = {0, pad_size, 0}
    rest_axes_pad = List.duplicate({0, 0, 0}, Nx.rank(tensor) - 1)
    Nx.pad(tensor, Nx.tensor(0, type: Nx.type(tensor)), [first_axis_pad | rest_axes_pad])
  end
end
