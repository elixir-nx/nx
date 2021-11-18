defprotocol Nx.Container do
  @moduledoc """
  A protocol that teaches Nx how to traverse custom data
  structures in search for tensors.

  This protocol can be automatically derived for custom
  data structures with:

      @derive {Nx.Container, [:field_name, :other_field]}

  The second argument is a list of fields that contains
  tensors.
  """

  @doc """
  Traverse receives a data structure, an accumulator,
  and a function that receives a tensor and the accumulator
  for each tensor in the data
  """
  @spec traverse(t(), acc, (Nx.Tensor.t(), acc -> {term(), acc})) :: acc when acc: term()
  def traverse(data, acc, fun)
end

defimpl Nx.Container, for: Tuple do
  def traverse(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.map_reduce(acc, fun)
    |> then(fn {list, acc} -> {List.to_tuple(list), acc} end)
  end
end

defimpl Nx.Container, for: Map do
  def traverse(map, acc, fun) do
    map
    |> Enum.map_reduce(acc, fn {k, v} ->
      {v, acc} = fun.(v, acc)
      {{k, v}, acc}
    end)
    |> then(fn {list, acc} -> {Map.new(list), acc} end)
  end
end

defimpl Nx.Container, for: [Nx.Tensor, Integer, Float] do
  def traverse(tensor, acc, fun) do
    fun.(tensor, acc)
  end
end

defimpl Nx.Container, for: Any do
  defmacro __deriving__(module, struct, fields) do
    for field <- fields do
      unless Map.has_key?(struct, field) do
        raise ArgumentError,
              "cannot derive Nx.Container for struct #{inspect(module)} " <>
                "because it does not have field #{inspect(field)}"
      end
    end

    field_vars =
      for field <- fields do
        {field, Macro.var(field, __MODULE__)}
      end

    updates =
      for {_, var} <- field_vars do
        quote do
          {unquote(var), var!(acc)} = var!(fun).(unquote(var), var!(acc))
        end
      end

    quote do
      defimpl Nx.Container, for: unquote(module) do
        def traverse(%{unquote_splicing(field_vars)} = struct, var!(acc), var!(fun)) do
          unquote_splicing(updates)
          {%{struct | unquote_splicing(field_vars)}, var!(acc)}
        end
      end
    end
  end

  def traverse(data, _acc, _fun) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: data,
      description: "TODO"
  end
end
