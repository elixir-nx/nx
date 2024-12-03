defmodule Nx.Defn.ShardingCompiler.ShardExecution.OutputCollector do
  use GenServer

  alias Nx.Defn.ShardingCompiler.ShardRegistry
  alias Nx.Defn.ShardingCompiler.ShardExecution
  alias Nx.Defn.ShardingCompiler.Passes.ShardPropagation

  alias Nx.Tensor, as: T

  def init([expr, previous_stage_id, listener_pid, is_sharded]) do
    {expr_tuple, unwrap_result} =
      if is_tuple(expr) do
        {expr, false}
      else
        {{expr}, true}
      end

    data_sections_by_index =
      if is_sharded do
        expr_tuple
        |> Tuple.to_list()
        |> Enum.with_index(fn expr, idx ->
          sections =
            for {starts, data_section_id} <- starts_and_data_section_ids(expr) do
              {data_section_id, starts, nil}
            end

          {idx, sections}
        end)
      else
        Map.new(0..(tuple_size(expr_tuple) - 1), fn idx -> {idx, [{:unsharded, nil, nil}]} end)
      end

    Process.send_after(self(), :collect_data, 0)

    {:ok,
     %{
       listener_pid: listener_pid,
       expr_tuple: expr_tuple,
       previous_stage_id: previous_stage_id,
       unwrap_result: unwrap_result,
       data_sections_by_index: data_sections_by_index,
       output: nil,
       is_sharded: is_sharded
     }}
  end

  def start_link(sharded_expr, previous_stage_id, listener_pid, is_sharded) do
    GenServer.start_link(
      __MODULE__,
      [sharded_expr, previous_stage_id, listener_pid, is_sharded],
      name: via_tuple(sharded_expr)
    )
  end

  defp via_tuple(expr) do
    expr_id =
      if is_tuple(expr) do
        expr
        |> Tuple.to_list()
        |> Enum.map(& &1.data.id)
      else
        expr.data.id
      end

    {:via, Registry, {ShardRegistry, {:output, expr_id}}}
  end

  def handle_call(:get, _from, state) do
    case state.output do
      nil -> {:reply, {:error, :pending}, state}
      data -> {:reply, {:ok, data}, state}
    end
  end

  def handle_info(:collect_data, state) do
    data_sections_by_index =
      Map.new(state.data_sections_by_index, fn {idx, data_sections} ->
        data_sections =
          for {data_section_id, starts, nil} <- data_sections do
            case ShardExecution.get(state.previous_stage_id, idx, data_section_id) do
              {:ok, data} ->
                {data_section_id, starts, data}

              {:error, :pending} ->
                {data_section_id, starts, nil}
            end
          end

        {idx, data_sections}
      end)

    finished =
      Enum.all?(data_sections_by_index, fn {_idx, data_sections} ->
        Enum.all?(data_sections, fn {_, _, data} -> not is_nil(data) end)
      end)

    output =
      if finished do
        out_list = produce_output(state.expr_tuple, data_sections_by_index, state.is_sharded)

        if state.unwrap_result do
          [out] = out_list
          out
        else
          List.to_tuple(out_list)
        end
      end

    if output do
      Process.send_after(self(), :notify_listener, 0)
    else
      Process.send_after(self(), :collect_data, 0)
    end

    {:noreply, %{state | output: output, data_sections_by_index: data_sections_by_index}}
  end

  def handle_info(:notify_listener, state) do
    send(state.listener_pid, {__MODULE__, :done, self(), state.output})
    {:noreply, state}
  end

  defp starts_and_data_section_ids(%T{data: %ShardPropagation{shards: shards}}) do
    shards
    |> Enum.sort_by(fn {axis, _} -> axis end)
    |> Enum.map(fn {axis, shard} -> {shard, axis} end)
    |> cartesian_product()
    |> Enum.map(fn sections ->
      starts =
        Enum.map(sections, fn {shard, _axis} -> shard.start end)

      data_section_id = Enum.map(sections, fn {shard, _axis} -> shard.id end)

      {starts, data_section_id}
    end)
  end

  defp starts_and_data_section_ids(%T{shape: shape, data: %Nx.Defn.Expr{id: id}}) do
    [List.duplicate(0, tuple_size(shape)), :unsharded]
  end

  defp cartesian_product([{data, meta} | rest]) do
    for x <- data, y <- cartesian_product(rest), do: [{x, meta} | y]
  end

  defp cartesian_product([]), do: [[]]

  defp produce_output(_expr_tuple, data_sections_by_index, false) do
    data_sections_by_index
    |> Enum.sort_by(fn {idx, _} -> idx end)
    |> Enum.map(fn {_idx, [{_, _, data}]} -> data end)
  end

  defp produce_output(expr_tuple, data_sections_by_index, true) do
    Enum.map(data_sections_by_index, fn {idx, data_sections} ->
      hole_template = elem(expr_tuple, idx)
      hole = Nx.broadcast(Nx.tensor(0, type: hole_template.type), hole_template.shape)

      data =
        Enum.reduce(data_sections, hole, fn {_, starts, data}, acc ->
          Nx.put_slice(acc, starts, data)
        end)

      {idx, data}
    end)
    |> Enum.sort_by(fn {idx, _} -> idx end)
    |> Enum.map(fn {_, data} -> data end)
  end
end
