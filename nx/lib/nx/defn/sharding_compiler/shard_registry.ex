defmodule Nx.Defn.ShardingCompiler.ShardRegistry do
  @table_name __MODULE__

  def init do
    :ets.new(@table_name, [:named_table, :set, :public])
    :ok
  end

  def register_producer({stage_id, index, data_section_id}) do
    :ets.insert(@table_name, {{stage_id, index, data_section_id}, :pending})
    :ok
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
end
