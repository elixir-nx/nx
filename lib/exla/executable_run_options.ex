defmodule Exla.ExecutableRunOptions do
  defstruct allocator: nil,
            device_ordinal: -1,
            device_assignment: nil,
            stream: nil,
            intra_op_thread_pool: nil,
            execution_profile: nil,
            rng_seed: nil,
            launch_id: 0,
            host_to_device_stream: nil,
            then_execute_function: nil,
            run_id: nil,
            gpu_executable_run_options: nil
end
