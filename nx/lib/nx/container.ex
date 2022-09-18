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

  > **Careful!**: If you keep a field, its value will be part
  > of the `Nx.Defn` compiler cache key (i.e. therefore if you
  > give a struct with two different values for a kept field,
  > `Nx.Defn` will have to compile and cache it twice). You
  > must only keep fields that you are certain to be used inside
  > `defn` during compilation time.
  """

  @fallback_to_any true

  @doc """
  Traverses non-recursively tensors in a data structure with `acc` and `fun`.

  `fun` is invoked with each tensor or tensor container in the
  data structure plus an accumulator. It must return a two element
  tuple with the updated value and accumulator.

  This function returns the updated container and the accumulator.

  Given `fun` may receive containers, it is not recursive by default.
  See `Nx.Defn.Composite.traverse/3` for a recursive variant.
  """
  @spec traverse(t(), acc, (Nx.t() | Nx.Container.t(), acc -> {Nx.t() | Nx.Container.t(), acc})) ::
          acc
        when acc: term()
  def traverse(data, acc, fun)

  @doc """
  Reduces non-recursively tensors in a data structure with `acc` and `fun`.

  `fun` is invoked with each tensor or tensor container in the
  data structure plus an accumulator. It must return the new
  accumulator.

  This function the final accumulator.

  Given `fun` may receive containers, it is not recursive by default.
  See `Nx.Defn.Composite.reduce/3` for a recursive variant.
  """
  @spec reduce(t(), acc, (Nx.t() | Nx.Container.t(), acc -> acc)) :: acc when acc: term()
  def reduce(data, acc, fun)
end

defimpl Nx.Container, for: Tuple do
  def traverse(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.map_reduce(acc, fun)
    |> then(fn {list, acc} -> {List.to_tuple(list), acc} end)
  end

  def reduce(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.reduce(acc, fun)
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

  def reduce(map, acc, fun) do
    map
    |> Map.to_list()
    |> Enum.sort()
    |> Enum.reduce(acc, fn {_, v}, acc -> fun.(v, acc) end)
  end
end

defimpl Nx.Container, for: Any do
  defmacro __deriving__(module, struct, options) do
    containers = Keyword.fetch!(options, :containers)
    keep = Keyword.get(options, :keep, [])

    container_pattern = Enum.map(containers, &field_var(struct, &1))
    keep_pattern = Enum.map(keep, &field_var(struct, &1))
    full_pattern = container_pattern ++ keep_pattern

    updates =
      for field <- containers do
        var = Macro.var(field, Nx.Container)

        quote do
          {unquote(var), acc} = fun.(unquote(var), acc)
        end
      end

    reduces =
      for field <- containers do
        var = Macro.var(field, Nx.Container)

        quote do
          acc = fun.(unquote(var), acc)
        end
      end

    return =
      struct
      |> Map.to_list()
      |> Keyword.drop(keep ++ containers)
      |> Macro.escape()
      |> Keyword.merge(full_pattern)

    quote do
      defimpl Nx.Container, for: unquote(module) do
        def traverse(%{unquote_splicing(full_pattern)} = struct, acc, fun) do
          unquote_splicing(updates)
          {%{unquote_splicing(return)}, acc}
        end

        def reduce(%{unquote_splicing(container_pattern)} = struct, acc, fun) do
          unquote_splicing(reduces)
          acc
        end
      end
    end
  end

  defp field_var(struct, field) do
    unless Map.has_key?(struct, field) do
      raise ArgumentError,
            "cannot derive Nx.Container for struct #{inspect(struct.__struct__)} " <>
              "because it does not have field #{inspect(field)}"
    end

    {field, Macro.var(field, Nx.Container)}
  end

  def traverse(data, _acc, _fun) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: data,
      description: "check the docs for Nx.Container for more information"
  end

  def reduce(data, _acc, _fun) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: data,
      description: "check the docs for Nx.Container for more information"
  end
end
