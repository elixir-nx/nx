defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Defn.ShardingCompiler.ShardingBackend
  alias Nx.Defn.ShardingCompiler.ShardingBackend.AxisConfig
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @behaviour Nx.Defn.Compiler

  @impl true
  def __jit__(key, vars, fun, args, opts) do
    [args] = args

    sharded_args =
      Enum.zip_with(args, opts[:sharding_config], fn arg, config ->
        fn -> ShardingBackend.shard(arg.(), config) end
      end)

    [%T{shape: shape, type: type, data: %ShardingBackend{sharding_config: output_config}}] =
      __compile__(key, vars, fun, opts).([sharded_args])

    slices =
      Enum.with_index(output_config, fn
        nil, axis -> {axis, [..]}
        %AxisConfig{slices: slices}, axis -> {axis, slices}
      end)

    shards = cartesian_product(slices)

    sharded_args =
      for slices <- shards do
        for lazy_input <- args do
          input = lazy_input.()

          slices = Enum.sort(slices)

          starts =
            for {_axis, start.._//1} <- slices do
              start
            end

          lengths =
            for {axis, start..finish//1} <- slices do
              axis_size = Nx.axis_size(input, axis)

              cond do
                axis_size == 1 ->
                  1

                finish == -1 ->
                  axis_size - start

                true ->
                  finish - start + 1
              end
            end

          {starts, Nx.slice(input, starts, lengths)}
        end
      end

    result =
      for instance <- sharded_args do
        args = Enum.map(instance, &elem(&1, 1))
        starts = Enum.map(instance, &elem(&1, 0))
        compiled_fun = Nx.Defn.Evaluator.__compile__(key, args, fun, [])

        {
          [args],
          fn args ->
            [res] =
              compiled_fun.([
                Enum.map(args, fn arg ->
                  fn -> arg end
                end)
              ])

            res
          end,
          fn result, acc ->
            Nx.put_slice(acc, hd(starts), result)
          end
        }
      end

    output_holder = Nx.iota(shape, type: type)

    [{output_holder, result}]
  end

  defp cartesian_product([{axis, first} | rest]) do
    for x <- first, y <- cartesian_product(rest), do: [{axis, x} | y]
  end

  defp cartesian_product([]), do: [[]]

  @impl true
  def __compile__(key, vars, fun, opts) do
    Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
  end

  def build_comprehension(slices) do
    args = Macro.generate_arguments(length(slices), nil)

    generators =
      Enum.zip_with(args, slices, fn arg, slice ->
        quote do
          unquote(arg) <- unquote(Macro.escape(slice))
        end
      end)

    {fun, []} =
      Code.eval_quoted(
        quote do
          fn arg ->
            for unquote_splicing(generators) do
              arg[unquote(args)]
            end
          end
        end
      )

    fun
  end

  def compile(fun, args, shard_args) do
    %{data: %{sharding_config: output_config}} = apply(fun, shard_args)

    slices =
      Enum.map(output_config, fn
        nil -> [..]
        %{slices: slices} -> slices
      end)

    slicer = build_comprehension(slices)

    shards = Enum.map(args, slicer)

    funs =
      Enum.zip_with(shards, fn args ->
        Nx.Defn.compile(fun, args, compiler: Nx.Defn.Evaluator)
      end)

    {shards, funs}
  end
end
