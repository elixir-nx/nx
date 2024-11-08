defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  alias Nx.Defn.ShardingCompiler.Shard

  alias Nx.Defn.ShardingCompiler.Passes.ShardPropagation
  # alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter

  @behaviour Nx.Defn.Compiler

  @impl true
  def __jit__(key, vars, fun, args, opts) do
    opts =
      Keyword.validate!(opts, [
        :sharding_config,
        sharding_compiler: Nx.Defn.Evaluator,
        sharding_compiler_options: []
      ])

    [args] = args

    {%T{
       type: type,
       data: %ShardPropagation{
         shards: output_shards
       }
     }, parameter_ids_to_index,
     shape} =
      propagate_shards(vars, fun, opts[:sharding_config] || [])

    data_sections =
      output_shards |> Enum.sort_by(fn {axis, _} -> axis end) |> cartesian_product()

    # Find the parents for each data section
    # Group by inputs
    # For each input, sort the shards by axis
    # For each axis, find the minimum start and the maximum end (we need to test for slicing inside the code as well)
    # it might be the case where an axis is not present in the mapping. This means we need the full axis.

    result =
      for section <- data_sections do
        shards_by_input_id =
          section
          |> Enum.flat_map(fn {_axis, shard} ->
            get_root_parents(shard)
          end)
          |> Enum.group_by(fn shard -> shard.input_id end)

        inputs_by_index =
          parameter_ids_to_index
          |> Enum.sort_by(fn {_id, idx} -> idx end)
          |> Enum.map(fn {id, idx} -> {id, Enum.fetch!(args, idx)} end)

        sliced_inputs =
          for {input_id, input_fn} <- inputs_by_index do
            input = input_fn.()
            shards = shards_by_input_id[input_id]
            shards_by_axis = Enum.group_by(shards, & &1.axis)

            {_, _, starts_reverse, lengths_reverse} =
              Enum.reduce(Tuple.to_list(input.shape), {shards_by_axis, 0, [], []}, fn
                axis_size, {shards_by_axis, axis, starts, lengths} ->
                  {shards, shards_by_axis} = Map.pop(shards_by_axis, axis)

                  {starts, lengths} =
                    if shards do
                      min_start = Enum.min(Enum.map(shards, & &1.start))
                      max_end = Enum.max(Enum.map(shards, &(&1.start + &1.length - 1)))

                      starts = [min_start | starts]
                      lengths = [max_end - min_start + 1 | lengths]
                      {starts, lengths}
                    else
                      starts = [0 | starts]
                      lengths = [axis_size | lengths]
                      {starts, lengths}
                    end

                  {shards_by_axis, axis + 1, starts, lengths}
              end)

            starts = Enum.reverse(starts_reverse)
            lengths = Enum.reverse(lengths_reverse)

            Nx.slice(input, starts, lengths)
          end

        {out_starts, []} =
          Enum.map_reduce(0..(tuple_size(shape) - 1)//1, section, fn
            axis, [{axis, shard} | shards] ->
              {shard.start, shards}

            _axis, shards ->
              {0, shards}
          end)

        caster_fn = fn result, acc ->
          Nx.put_slice(acc, out_starts, result)
        end

        sharding_compiler = opts[:sharding_compiler]
        sharding_compiler_options = opts[:sharding_compiler_options]

        vars =
          Enum.with_index(sliced_inputs, fn arg, idx ->
            arg
            |> Expr.parameter(:root, idx)
          end)

        compiled_fun =
          sharding_compiler.__compile__({key, section}, vars, fun, sharding_compiler_options)

        shard_fn = fn [args] ->
          [res] =
            compiled_fun.([
              Enum.map(Tuple.to_list(args), fn arg ->
                fn -> arg end
              end)
            ])

          res
        end

        {[List.to_tuple(sliced_inputs)], shard_fn, caster_fn}
      end

    output_holder = Nx.iota(shape, type: type)
    [{output_holder, result}]
  end

  defp cartesian_product([{axis, first} | rest]) do
    for x <- first, y <- cartesian_product(rest), do: [{axis, x} | y]
  end

  defp cartesian_product([]), do: [[]]

  @impl true
  def __compile__(_key, _vars, _fun, _opts) do
    raise "Not implemented yet"
  end

  def propagate_shards(vars, fun, sharding_config) do
    expr = fun.(vars)

    tensor_shardings =
      sharding_config
      |> Enum.zip_with(vars, fn config, var ->
        Shard.from_config(var, config)
      end)
      |> Enum.with_index(fn x, idx -> {idx, x} end)
      |> Map.new()

    {container, _cache, state} = ShardPropagation.traverse(expr, tensor_shardings)

    {container, state.parameter_ids_to_index, expr.shape}
  end

  @impl true
  def __partitions_options__(_keyword) do
    raise "__partitions_options__ not supported"
  end

  @impl true
  def __to_backend__(_keyword) do
    raise "__to_backend__ not supported"
  end

  def init(opts), do: opts

  defp get_root_parents(shard, acc \\ [])

  defp get_root_parents(%Shard{parents: []} = shard, acc), do: List.flatten([shard | acc])

  defp get_root_parents(%Shard{parents: parents}, acc) do
    Enum.reduce(parents, acc, &get_root_parents/2)
    |> List.flatten()
  end
end
