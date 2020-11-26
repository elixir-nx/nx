defmodule Exla.Defn do
  @moduledoc false
  # The Exla compiler for defn mixes runtime with compile-time
  # execution. The goal of the compiler pass is to convert all
  # special forms and Nx calls to calls to this module. All
  # calls are prefixed with either `sf_` or `nx_`.
  #
  # This module then builds the Exla AST as part of its code
  # evaluation process by invoking said functions.

  ## Builder and computations

  def sf_cached_def(module, name_arity, args, options, fun) do
    cache_args = for arg <- args, do: nx_to_cache_key!(arg)
    buffers = for arg <- args, do: nx_to_buffer(arg)
    cache_key = {module, name_arity, cache_args}

    executable =
      Exla.LockedCache.run(cache_key, fn ->
        shapes = Enum.map(buffers, & &1.shape)
        result = apply(fun, shapes)
        computation = Exla.Builder.build(result)
        client = Exla.Client.fetch!(Keyword.get(options, :client, :default))
        executable = Exla.Client.compile(client, computation, shapes)
        :persistent_term.put(cache_key, executable)
        executable
      end)

    # TODO: Pass options
    executable
    |> Exla.Executable.run(buffers, [])
    |> buffer_to_nx()
  end

  def sf_builder(name) do
    Exla.Builder.new(name)
  end

  def sf_parameter(builder, pos, shape, name) do
    Exla.Op.parameter(builder, pos, shape, name)
  end

  ## Nx <-> Exla.Buffer
  # TODO: What to do when the tensor data is not a binary?

  defp buffer_to_nx(%Exla.Buffer{ref: nil, data: data, shape: shape}) do
    %Nx.Tensor{data: data, type: shape.dtype, shape: shape.dims}
  end

  defp nx_to_buffer(%Nx.Tensor{data: data, type: type, shape: shape})
       when is_bitstring(data) do
    Exla.Buffer.buffer(data, Exla.Shape.make_shape(type, shape))
  end

  defp nx_to_buffer(number) when is_integer(number) do
    Exla.Buffer.buffer(<<number::64-native>>, Exla.Shape.make_shape({:s, 64}, {}))
  end

  defp nx_to_buffer(number) when is_float(number) do
    Exla.Buffer.buffer(<<number::float-64-native>>, Exla.Shape.make_shape({:f, 64}, {}))
  end

  defp nx_to_cache_key!(number) when is_integer(number), do: {{:s, 64}, {}}
  defp nx_to_cache_key!(number) when is_float(number), do: {{:f, 64}, {}}
  defp nx_to_cache_key!(%Nx.Tensor{} = t), do: {t.type, t.shape}

  defp nx_to_cache_key!(other) do
    raise ArgumentError,
          "defn functions expects either numbers or %Nx.Tensor{} as arguments, " <>
            "got: #{inspect(other)}"
  end

  ## Special forms

  ## Operators

  def nx_add(builder, left, right) do
    left_shape = Exla.Shape.get_shape(left)
    right_shape = Exla.Shape.get_shape(right)
    dims = broadcast_dimensions(left_shape.dims, right_shape.dims)

    left = to_operator(builder, left)
    right = to_operator(builder, right)
    Exla.Op.add(left, right, dims)
  end

  def nx_divide(builder, left, right) do
    left = to_operator(builder, left)
    right = to_operator(builder, right)
    Exla.Op.div(left, right)
  end

  def nx_exp(builder, op) do
    Exla.Op.exp(to_operator(builder, op))
  end

  def nx_sum(builder, op) do
    op = to_operator(builder, op)
    op_shape = Exla.Shape.get_shape(op)
    reduction_shape = Exla.Shape.make_shape(op_shape.dtype, {})

    # Build the anonymous function
    unique = System.unique_integer([:positive])
    sub_builder = Exla.Builder.new(builder, builder.name <> "-sum-" <> Integer.to_string(unique))
    a = Exla.Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Exla.Op.parameter(sub_builder, 1, reduction_shape, "b")
    add = Exla.Op.add(a, b)
    reduction = Exla.Builder.build(add)

    init_value = Exla.Op.constant_r0(builder, 0, reduction_shape.dtype)
    Exla.Op.reduce(op, init_value, reduction, all_dimensions(op_shape.dims))
  end

  defp to_operator(_builder, %Exla.Op{} = op), do: op

  defp to_operator(_builder, number) when is_number(number) do
    # TODO: fix me
    raise "not yet support. change constant_r0 to allow both integers and floats and custom shapes"
  end

  # Converts {3, 255, 255} into {0, 1, 2}
  defp all_dimensions(dims), do: List.to_tuple(all_dimensions(0, tuple_size(dims)))
  defp all_dimensions(i, n) when i < n, do: [i | all_dimensions(i + 1, n)]
  defp all_dimensions(_, _), do: []

  defp broadcast_dimensions(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}

    min_size = tuple_size(min)
    max_size = tuple_size(max)

    if min_size == max_size do
      all_dimensions(min)
    else
      min_ordered = shape_to_ranked_ordered_list(min, min_size)
      max_ordered = shape_to_ranked_ordered_list(max, max_size)

      # The initial pass maps all dimensions of equal size and
      # all degenerate dimensions on the lower-ranked tuple to
      # the same dimension on the higher-ranked tuple.
      case broadcast_dimensions(min_ordered, max_ordered, 0, min_size, []) do
        {pending, acc} ->
          # If we still didn't fullfil the broadcast dimensions, it is
          # because the higher-ranked tuple has generate dimensions.
          # So we map the pending dimensions to the highest dimensions
          # on the higher-ranked tuple.
          List.to_tuple(Enum.reverse(count_down(pending, max_size - pending, acc)))

        :error ->
          raise ArgumentError,
                "cannot broadcast tensor of dimensions #{inspect(left)} to #{inspect(right)}"
      end
    end
  end

  defp count_down(0, _n, acc), do: acc
  defp count_down(i, n, acc), do: count_down(i - 1, n + 1, [n | acc])

  defp broadcast_dimensions([dim | left], [dim | right], n, pending, acc),
    do: broadcast_dimensions(left, right, n + 1, pending - 1, [n | acc])

  defp broadcast_dimensions([1 | left], [_rdim | right], n, pending, acc),
    do: broadcast_dimensions(left, right, n + 1, pending - 1, [n | acc])

  defp broadcast_dimensions([_ldim | left], [1 | right], n, pending, acc),
    do: broadcast_dimensions(left, right, n + 1, pending, acc)

  defp broadcast_dimensions([_ | _], [_ | _], _n, _pending, _acc),
    do: :error

  defp broadcast_dimensions([], _right, _n, pending, acc),
    do: {pending, acc}

  defp shape_to_ranked_ordered_list(_tuple, 0),
    do: []

  defp shape_to_ranked_ordered_list(tuple, size),
    do: [:erlang.element(size, tuple) | shape_to_ranked_ordered_list(tuple, size - 1)]

  ## Callback

  def __compile__(_kind, _meta, name, args, ast, options) do
    state = %{}

    {ast, _state} = traverse(ast, state)
    arity = length(args)
    builder = "#{name}/#{arity}"
    shapes = Macro.generate_arguments(arity, __MODULE__)

    quote do
      Exla.Defn.sf_cached_def(
        __MODULE__,
        {unquote(name), unquote(arity)},
        unquote(args),
        unquote(options),
        fn unquote_splicing(shapes) ->
          builder = Exla.Defn.sf_builder(unquote(builder))
          unquote_splicing(args_to_parameters(args, shapes))
          unquote(ast)
        end
      )
    end
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    {args, state} = traverse(args, state)
    {{{:., dot_meta, [__MODULE__, :"nx_#{name}"]}, meta, [quote(do: builder) | args]}, state}
  end

  defp traverse({name, meta, args}, state) do
    {args, state} = traverse(args, state)
    {{name, meta, args}, state}
  end

  defp traverse({left, right}, state) do
    {left, state} = traverse(left, state)
    {right, state} = traverse(right, state)
    {{left, right}, state}
  end

  defp traverse(list, state) when is_list(list) do
    Enum.map_reduce(list, state, &traverse/2)
  end

  defp traverse(other, state) do
    {other, state}
  end

  defp args_to_parameters(args, shapes) do
    for {{var, shape}, index} <- Enum.with_index(Enum.zip(args, shapes)) do
      name = var_to_parameter_name(var)

      quote do
        unquote(var) =
          Exla.Defn.sf_parameter(builder, unquote(index), unquote(shape), unquote(name))
      end
    end
  end

  defp var_to_parameter_name({var, meta, ctx}) when is_atom(var) and is_atom(ctx) do
    "#{var}@#{Keyword.fetch!(meta, :counter)}"
  end
end
