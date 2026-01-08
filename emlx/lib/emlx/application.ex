defmodule EMLX.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      EMLX.Cleaner
    ]

    opts = [strategy: :one_for_one, name: EMLX.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
