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
    :fetched_inputs
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
    Process.send_after(self(), 0, :initialize)

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
    GenServer.start_link(__MODULE__, args)
  end

  def handle_info(:initialize, state) do
    ShardRegistry.register_producer(
      {state.stage.id, state.output_entry_index, state.output_data_section_id}
    )

    {:noreply, state}
  end

  def handle_info(:fetch_inputs, state) do
    state =
      for {arg_idx, {arg_id, data_section_id}} <- state.input_data_sections,
          is_nil(state.fetched_inputs[arg_id]),
          reduce: state do
        state ->
          {stage_id, stage_idx} = state.stage.argument_sources[arg_id]

          key = {stage_id, stage_idx, data_section_id}

          case ShardRegistry.get(key) do
            {:ok, data} ->
              put_in(state.fetched_inputs[arg_id], {arg_idx, data})

            {:error, :pending} ->
              state
          end
      end

    if Enum.any?(state.fetched_inputs, fn {_arg_id, data} -> is_nil(data) end) do
      Process.send_after(self(), 20, :fetch_inputs)
      {:noreply, state}
    else
      compute(state)
    end
  end

  defp compute(state) do
    {:noreply, state}
  end
end
