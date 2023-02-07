defmodule Nx.Application do
  @moduledoc false
  use Application

  def start(_type, _args) do
    children = [
      %{id: Nx.Serving.PG, start: {:pg, :start_link, [Nx.Serving.PG]}}
    ]

    Supervisor.start_link(children, strategy: :one_for_all)
  end
end
