defmodule Nx.Defn.ShardingCompiler.ShardRegistry do
  use GenServer
  @table_name __MODULE__

  def init(_) do
    :ets.new(@table_name, [:named_table, :set, :public])
    pid_to_key_table = :ets.new(:pid_to_key, [:bag, :protected])
    {:ok, %{pid_to_key_table: pid_to_key_table}}
  end

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def register_producer(key) do
    GenServer.call(__MODULE__, {:register, key})
  end

  def get({stage_id, index, data_section_id}) do
    case :ets.lookup(@table_name, {stage_id, index, data_section_id}) do
      [{{^stage_id, ^index, ^data_section_id}, {:ready, tensor}}] ->
        {:ok, tensor}

      [{{^stage_id, ^index, ^data_section_id}, :pending}] ->
        {:error, :pending}

      _ ->
        {:error, :not_found}
    end
  end

  def handle_call({:register_producer, key}, pid, state) do
    :ets.insert(@table_name, {key, :pending})
    :ets.insert(state.pid_to_key_table, {pid, key})
    Process.monitor(pid)
    {:reply, :ok, state}
  end

  def handle_info({:DOWN, _ref, :process, producer_pid, _reason}, state) do
    for {^producer_pid, key} <- :ets.lookup(state.pid_to_key_table, producer_pid) do
      :ets.delete(@table_name, key)
    end

    true = :ets.delete(state.pid_to_key_table, producer_pid)

    {:noreply, state}
  end
end
