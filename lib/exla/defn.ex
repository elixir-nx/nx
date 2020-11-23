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

  # TODO: Implement caching
  def sf_cached_def(module, name_arity, args, options, fun) do
    buffers = for arg <- args, do: elixir_to_buffers(arg)
    cache_args = for arg <- args, do: elixir_to_cache_key(arg)
    cache_key = {module, name_arity, cache_args}

    executable =
      case :persistent_term.get(cache_key, :undefined) do
        :undefined ->
          shapes = Enum.map(buffers, & &1.shape)
          result = apply(fun, shapes)
          computation = Exla.Builder.build(result)
          # TODO: Change all platform: :host to platform: :host and read options
          client = Exla.Client.create_client(platform: Keyword.get(options, :platform, :host))
          # TODO: Make shapes a list
          executable = Exla.Client.compile(client, computation, List.to_tuple(shapes))
          :persistent_term.put(cache_key, executable)
          executable

        executable ->
          executable
      end

    # TODO: Pass options
    # TODO: Make buffers a list
    # TODO: Convert this back to a tensor
    Exla.Executable.run(executable, List.to_tuple(buffers), [])
  end

  def sf_builder(name) do
    Exla.Builder.new(name)
  end

  def sf_parameter(builder, pos, shape, name) do
    Exla.Op.parameter(builder, pos, shape, name)
  end

  # TODO: Tensor type is hardcoded, we need to unify them
  defp elixir_to_buffers(number) when is_integer(number) do
    Exla.Buffer.buffer(<<number::64-native>>, Exla.Shape.make_shape(:int64, {}))
  end

  defp elixir_to_buffers(number) when is_float(number) do
    Exla.Buffer.buffer(<<number::float-64-native>>, Exla.Shape.make_shape(:float64, {}))
  end

  # TODO: Tensor type is hardcoded, we need to unify them
  # TODO: What to do when the tensor data is not a binary?
  defp elixir_to_buffers(%Nx.Tensor{data: data, type: _type, shape: shape})
       when is_bitstring(data) do
    Exla.Buffer.buffer(data, Exla.Shape.make_shape(:float64, shape))
  end

  defp elixir_to_cache_key(number) when is_integer(number), do: {{:i, 64}, {}}
  defp elixir_to_cache_key(number) when is_float(number), do: {{:f, 64}, {}}
  defp elixir_to_cache_key(%Nx.Tensor{} = t), do: {t.type, t.shape}

  ## Special forms


  ## Operators

  def nx_add(builder, left, right) do
    left = to_operator(builder, left)
    right = to_operator(builder, right)
    Exla.Op.add(left, right)
  end

  def nx_divide(builder, left, right) do
    left = to_operator(builder, left)
    right = to_operator(builder, right)
    Exla.Op.div(left, right)
  end

  def nx_exp(builder, op) do
    Exla.Op.exp(to_operator(builder, op))
  end

  # TODO: is unique integer the best way to go about this?
  def nx_sum(builder, op) do
    # TODO: Use operator get shape instead
    op = to_operator(builder, op)
    op_shape = Exla.Shape.make_shape(:float64, {4})
    reduction_shape = Exla.Shape.make_shape(op_shape.dtype, {})

    # Build the anonymous function
    unique = System.unique_integer([:positive])
    sub_builder = Exla.Builder.new(builder, builder.name <> "-sum-" <> Integer.to_string(unique))
    a = Exla.Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Exla.Op.parameter(sub_builder, 1, reduction_shape, "b")
    add = Exla.Op.add(a, b)
    reduction = Exla.Builder.build(add)

    init_value = Exla.Op.zero(builder, reduction_shape.dtype)
    Exla.Op.reduce(op, init_value, reduction, all_dimensions(op_shape))
  end

  # Converts {3, 255, 255} into {0, 1, 2}
  defp all_dimensions(shape) do
    0..tuple_size(shape.dims)
    |> Enum.to_list()
    |> tl()
    |> Enum.map(& &1 - 1)
    |> List.to_tuple()
  end

  defp to_operator(_builder, %Exla.Op{} = op), do: op

  defp to_operator(_builder, number) when is_number(number) do
    raise "not yet support. change constant_r0 to allow both integers and floats and custom shapes"
  end

  ## Callback

  def __compile__(_kind, _meta, name, args, ast, options) do
    # TODO: Build lock mechanism
    # TODO: Should we store the name in the sf_builder?
    state = %{
      computation_counter: 0
    }

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
      end)
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
