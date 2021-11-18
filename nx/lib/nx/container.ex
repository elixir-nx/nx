defprotocol Nx.Container do
  @moduledoc """
  A protocol that teaches Nx how to traverse data structures.

  `Nx` and `defn` expect the arguments to be numbers, tensors,
  or one of the following composite data types:

    1. tuples of numbers/tensors
    2. maps of any key with numbers/tensors as values
    3. any struct that implements `Nx.Container`

  If you need to pass additional values, you can implement
  or derive this protocol. For example:

      @derive {Nx.Container,
               containers: [:field_name, :other_field]}
      defstruct [:field_name, :other_fields, ...]

  The `:containers` option is required and it must specify a
  list of fields that contains tensors. Inside `defn`, the
  container fields will be automatically converted to tensor
  expressions. All other fields will be reset to their default
  value, unless you explicitly declare them to be kept:

      @derive {Nx.Container,
               containers: [:field_name, :other_field],
               keep: [:another_field]}
      defstruct [:field_name, :other_fields, ...]

  If you keep a field, its value will be part of the `Nx.Defn`
  compiler cache key (i.e. therefore if you give a struct with
  two different values for a kept field, `Nx.Defn` will have
  to compile and cache it twice).
  """

  @fallback_to_any true

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
    |> Map.to_list()
    |> Enum.sort()
    |> Enum.map_reduce(acc, fn {k, v}, acc ->
      {v, acc} = fun.(v, acc)
      {{k, v}, acc}
    end)
    |> then(fn {list, acc} -> {Map.new(list), acc} end)
  end
end

defimpl Nx.Container, for: Nx.Tensor do
  def traverse(tensor, acc, fun) do
    fun.(tensor, acc)
  end
end

defimpl Nx.Container, for: [Integer, Float] do
  def traverse(number, acc, fun) do
    fun.(Nx.to_tensor(number), acc)
  end
end

defimpl Nx.Container, for: Any do
  defmacro __deriving__(module, struct, options) do
    containers = Keyword.fetch!(options, :containers)
    keep = Keyword.get(options, :keep, [])

    pattern =
      for field <- containers ++ keep do
        unless Map.has_key?(struct, field) do
          raise ArgumentError,
                "cannot derive Nx.Container for struct #{inspect(module)} " <>
                  "because it does not have field #{inspect(field)}"
        end

        {field, Macro.var(field, __MODULE__)}
      end

    updates =
      for field <- containers do
        var = Macro.var(field, __MODULE__)

        quote do
          {unquote(var), var!(acc)} = var!(fun).(unquote(var), var!(acc))
        end
      end

    return = struct |> Map.to_list() |> Keyword.merge(pattern)

    quote do
      defimpl Nx.Container, for: unquote(module) do
        def traverse(%{unquote_splicing(pattern)} = struct, var!(acc), var!(fun)) do
          unquote_splicing(updates)
          {%{unquote_splicing(return)}, var!(acc)}
        end
      end
    end
  end

  def traverse(data, _acc, _fun) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: data,
      description: "check the docs for Nx.Container for more information"
  end
end
