defmodule Nx.Sharding.PartitionedExecutor.Supervisor do
  @moduledoc false

  use Supervisor

  alias Nx.Sharding.PartitionedExecutor.Function, as: F

  require Logger

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg)
  end

  def init(functions) do
    children =
      for %F{} = function <- functions do
        # TO-DO: mark these as transient when we know how to
        # identify that every process that depends on this
        # one directly has already run successfully
        Supervisor.child_spec({F, function}, id: {F, function.id})
      end

    Supervisor.init(children, strategy: :one_for_one)
  end
end
