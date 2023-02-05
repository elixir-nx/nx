defmodule Nx.Serving.Supervisor do
  @moduledoc false
  @behaviour Supervisor

  def init({key, payload, children, opts}) do
    :persistent_term.put(key, payload)
    Supervisor.init(children, opts)
  end
end