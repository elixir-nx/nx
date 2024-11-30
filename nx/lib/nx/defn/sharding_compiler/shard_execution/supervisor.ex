defmodule Nx.Defn.ShardingCompiler.ShardExecution.Supervisor do
  use Supervisor

  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage
  alias Nx.Defn.ShardingCompiler.Shard

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  def start_link(%Stage{} = stage) do
    Supervisor.start_link(__MODULE__, stage)
  end

  @impl true
  def init(stage) do
    children =
      for {output_entry_index, output_data_sections} <- output_data_sections(stage),
          {output_data_section_id, input_data_sections, output_starts, output_lengths} <-
            input_data_sections(stage.arguments, output_data_sections) do
        %{
          id: {stage.id, output_entry_index, output_data_section_id},
          start:
            {Nx.Defn.ShardingCompiler.ShardExecution, :start_link,
             [
               [
                 stage,
                 input_data_sections,
                 output_entry_index,
                 output_data_section_id,
                 output_starts,
                 output_lengths
               ]
             ]},
          restart: :permanent,
          type: :worker
        }
      end

    Supervisor.init(children, strategy: :one_for_one)
  end

  defp output_data_sections(%Stage{expr: expr}) do
    if is_tuple(expr) do
      expr
      |> Tuple.to_list()
      |> Enum.with_index(fn expr, idx -> {idx, output_data_sections_for_expr(expr)} end)
    else
      [{0, output_data_sections_for_expr(expr)}]
    end
  end

  defp output_data_sections_for_expr(%T{data: %Expr{op: :metadata, args: [_, %{shards: shards}]}}) do
    shards
    |> Enum.sort_by(fn {axis, _} -> axis end)
    |> Enum.map(fn {axis, shard} -> {shard, axis} end)
    |> cartesian_product()
    |> Enum.map(fn sections ->
      {starts, lengths} =
        sections
        |> Enum.map(fn {shard, _axis} -> {shard.start, shard.length} end)
        |> Enum.unzip()

      data_section_id = Enum.map(sections, fn {shard, _axis} -> shard.id end)

      roots =
        Enum.map(sections, fn {shard, axis} -> {axis, get_root_parents(shard)} end)

      {data_section_id, {roots, starts, lengths}}
    end)
  end

  defp input_data_sections(arguments, output_data_sections) do
    for {data_section_id, {output_roots_by_dim, starts, lengths}} <- output_data_sections do
      arg_sections =
        for {arg_id, arg} <- arguments do
          if arg.shape == {} do
            %T{data: %Expr{op: :metadata, args: [param, _]}} = arg
            %T{data: %Expr{op: :parameter, args: [arg_idx]}} = param

            # TODO: this is probably wrong
            {arg_idx, {arg_id, :scalar}}
          else
            %T{data: %Expr{op: :metadata, args: [param, %{shards: shards}]}} = arg
            %T{data: %Expr{op: :parameter, args: [arg_idx]}} = param

            require IEx

            shards_by_root =
              for {_axis, shards_for_axis} <- shards,
                  shard <- shards_for_axis,
                  root <- get_root_parents(shard),
                  into: %{} do
                {root.id, shard}
              end

            data_section_id_for_input =
              output_roots_by_dim
              |> Enum.flat_map(fn {_axis, roots} ->
                case Enum.filter(roots, &shards_by_root[&1.id]) do
                  [] -> []
                  shards -> Enum.map(shards, &{&1.axis, &1.id})
                end
              end)
              |> Enum.sort()
              |> Enum.uniq()
              |> Enum.map(fn {_axis, id} -> id end)

              dbg(data_section_id_for_input)
            if length(data_section_id_for_input) == tuple_size(arg.shape) do
              {arg_idx, {arg_id, data_section_id_for_input}}
            else
              require IEx
              IEx.pry()
              {arg_idx, {arg_id, :ignore}}
            end
          end
        end

      {data_section_id, arg_sections, starts, lengths}
    end
  end

  defp cartesian_product([{data, meta} | rest]) do
    for x <- data, y <- cartesian_product(rest), do: [{x, meta} | y]
  end

  defp cartesian_product([]), do: [[]]

  defp get_root_parents(shard, acc \\ [])

  defp get_root_parents(%Shard{parents: []} = shard, acc), do: List.flatten([shard | acc])

  defp get_root_parents(%Shard{parents: parents}, acc) do
    Enum.reduce(parents, acc, &get_root_parents/2)
    |> List.flatten()
  end
end
