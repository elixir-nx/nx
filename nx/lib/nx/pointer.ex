defmodule Nx.Pointer do
  @moduledoc """
  Represents a reference to a value in memory.

  Can represent either a pointer or an IPC handle.
  """

  @type t :: %Nx.Pointer{
          address: nil | non_neg_integer(),
          kind: :local | :ipc,
          data_size: pos_integer(),
          handle: nil | binary()
        }

  defstruct [:address, :kind, :data_size, :handle]
end
