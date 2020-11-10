defmodule Exla.Options.LocalClientOptions do
  defstruct platform: :host,
            number_of_replicas: 1,
            intra_op_parallelism_threads: -1,
            allowed_devices: nil
end
