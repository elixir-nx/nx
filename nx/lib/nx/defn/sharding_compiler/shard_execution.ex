defmodule Nx.Defn.ShardingCompiler.ShardExecution do
  # processes a single shard of an output entry, given the corresponding input data sections (1 per input)
  defstruct [
    :compiled_fun,
    :stage,
    :input_data_sections,
    :output_entry_index,
    :output_data_section_id,
    :output_starts,
    :output_lengths,
    :fetched_inputs,
    :output
  ]

  use GenServer

  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage
  alias Nx.Defn.ShardingCompiler.ShardRegistry

  def init(
        %Stage{} = stage,
        input_data_sections,
        output_entry_index,
        output_data_section_id,
        output_starts,
        output_lengths
      ) do
    Process.send_after(self(), :fetch_inputs, 0)

    fetched_inputs = Map.new(input_data_sections, fn {_idx, {arg_id, _}} -> {arg_id, nil} end)

    {:ok,
     %__MODULE__{
       stage: stage,
       input_data_sections: input_data_sections,
       output_entry_index: output_entry_index,
       output_data_section_id: output_data_section_id,
       output_starts: output_starts,
       output_lengths: output_lengths,
       fetched_inputs: fetched_inputs
     }}
  end

  def start_link(args) do
    GenServer.start_link(__MODULE__, args, name: via_tuple(args))
  end

  defp via_tuple([stage, _input_data_sections, output_entry_index, output_data_section_id]) do
    {:via, Registry, {ShardRegistry, {stage.id, output_entry_index, output_data_section_id}}}
  end

  def handle_info(:fetch_inputs, state) do
    state =
      for {arg_idx, {arg_id, data_section_id}} <- state.input_data_sections,
          is_nil(state.fetched_inputs[arg_id]),
          reduce: state do
        state ->
          {stage_id, stage_idx} = state.stage.argument_sources[arg_id]

          case get(stage_id, stage_idx, data_section_id) do
            {:ok, data} ->
              put_in(state.fetched_inputs[arg_id], {arg_idx, data})

            {:error, :pending} ->
              state
          end
      end

    if Enum.any?(state.fetched_inputs, fn {_arg_id, data} -> is_nil(data) end) do
      Process.send_after(self(), 10, :fetch_inputs)
      {:noreply, state}
    else
      state = compute(state)
      {:noreply, state}
    end
  end

  def get(stage_id, stage_idx, data_section_id) do
    key = {stage_id, stage_idx, data_section_id}

    case ShardRegistry.lookup(key) do
      {:ok, pid} -> GenServer.call(pid, :get)
      {:error, :pending} -> {:error, :pending}
    end
  end

  def handle_call(:get, _from, state) do
    result =
      case state.output do
        nil -> {:error, :pending}
        data -> {:ok, {data, state.output_starts, state.output_lengths}}
      end

    {:reply, result, state}
  end

  defp compute(state) do
    output = state.compiled_fun.(state.fetched_inputs)
    %{state | output: output}
  end
end
