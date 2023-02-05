defmodule Nx.Serving.Default do
  @moduledoc false
  @behaviour Nx.Serving

  @impl true
  def init(_type, fun, defn_options) do
    case fun.(defn_options) do
      batch_fun when is_function(batch_fun, 1) ->
        {:ok, batch_fun}

      other ->
        raise "anonymous function given to Nx.Serving.new/2 should return an AOT or " <>
                "JIT compiled function that expects one argument. Got: #{inspect(other)}"
    end
  end

  @impl true
  def handle_batch(batch, batch_fun) do
    {:execute, fn -> {batch_fun.(batch), :server_info} end, batch_fun}
  end
end
