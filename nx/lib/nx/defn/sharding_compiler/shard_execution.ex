defmodule Nx.Defn.ShardingCompiler.ShardExecution do
  defstruct [:compiled_fun, :stage_id, :input_data_sections, :output_data_sections]

  use GenServer

  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage

  def init(
        stage,
        input_data_sections,
        output_entry_index,
        output_data_section_id,
        output_starts,
        output_lengths
      ) do
    Process.send_after(self(), 0, :initialize)

    {:ok,
     %{
       stage: stage,
       input_data_sections: input_data_sections,
       output_entry_index: output_entry_index,
       output_data_section_id: output_data_section_id,
       output_starts: output_starts,
       output_lengths: output_lengths
     }}
  end

  def start_link(args) do
    GenServer.start_link(__MODULE__, args)
  end

  def handle_info(:initialize, state) do
    {:noreply, state}
  end

  def handle_info(:fetch_inputs, state) do
    inputs = Enum.map(state.data_sections, fn {_, shard} -> shard.input_id end)
    {:noreply, state, inputs}
  end
end
