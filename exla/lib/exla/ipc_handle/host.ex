defmodule EXLA.IPCHandle.Host do
  @moduledoc """
  Represents an IPC handle for sharing data between host OS processes.

  `:name` is the name of the shared memory object.
  `:fd` is the file descriptor of the shared memory object.
  `:size` is the size of the memory in bytes.
  """

  defstruct [:name, :fd, :size]
end
