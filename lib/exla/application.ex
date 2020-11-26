defmodule Exla.Application do
  @moduledoc false

  def start(_args, _type) do
    Supervisor.start_link([Exla.LockedCache], name: __MODULE__, strategy: :one_for_one)
  end
end
