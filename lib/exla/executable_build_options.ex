defmodule Exla.ExecutableBuildOptions do
  defstruct [
    :result_layout,
    device_ordinal: -1,
    debug_options: nil,
    device_allocator: nil,
    num_replicas: 1,
    num_partitions: 1,
    use_spmd_partitioning: false,
    duplicate_hlo: false,
    device_assignment: nil,
    alias_passthrough_params: false
  ]
end
