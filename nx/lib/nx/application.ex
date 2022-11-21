defmodule Nx.Application do
  @moduledoc false
  use Application

  def start(_type, _arg) do
    Supervisor.start_link([], strategy: :one_for_one)
  end
end
