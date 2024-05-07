defmodule EXLA.IPCHandle.CUDA do
  @moduledoc """
  Represents an IPC handle for sharing data allocated on a CUDA device between OS processes.

  `:handle` is a binary-encoded `cudaIpcMemHandle_t` that can be loaded
  at the native level via `memcpy` or similar functions.

  `:device_id` is the CUDA device ID that the memory is allocated on.

  `:size` is the size of the memory in bytes.
  """

  defstruct [:handle, :device_id, :size]
end
