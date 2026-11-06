defmodule EXLA.CustomCall.Spec do
  @moduledoc """
  Result of `EXLA.CustomCall.call/4` when lowering a tagged `Nx.block/4` to
  `stablehlo.custom_call`.

    * **`call_target_name`** — XLA FFI handler name (`call_target_name` on the op).

    * **`api_version`** — The custom call API version, default `4` (typed FFI). Targets
      that XLA lowers itself with the earlier ABI, such as the cuDNN fused attention
      targets, take `2`.

    * **`backend_config`** — Optional binary, default `nil`, emitted as the opaque string
      `backend_config` of the op. Only an `api_version` below 4 takes it, and then
      `attributes` must be empty. For example, the cuDNN targets read a JSON
      `GpuBackendConfig` from it.

    * **`attributes`** — Optional `{name, attr}` pairs, default `[]`, merged into
      the `backend_config` dictionary on `stablehlo.custom_call` (StableHLO’s name
      for that attribute). Each `name` must be a **BEAM binary** MLIR identifier; each
      `attr` must be a **BEAM binary** with valid MLIR attribute syntax for the RHS after
      `name = ` (for example `{"k", "42 : i64"}`). An empty list omits the dictionary
      from the op.

    * **`mlir_attributes`** — Optional `{name, attr}` pairs, default `[]`, emitted
      directly on `stablehlo.custom_call`. Use these for compiler-visible metadata such
      as layouts and sharding rules. Names and attributes use the same binary format as
      `attributes`. Names must be unique and cannot be `call_target_name`, `api_version`,
      or `backend_config`.

    * **`extra_results`** — Optional `{type, shape}` pairs, default `[]`, of results that
      the custom call has after the block outputs, such as a scratch workspace. Shapes can
      have zero-size axes. EXLA drops these results, so the block outputs stay the output
      of `Nx.block/4`.

    * **`operand_element_types`** — How operand SSA values are presented to the handler:

      * **`:default`** — use each lowered operand’s element type as produced from the
        block inputs. No extra converts.

      * **`[Nx.Type.t(), ...]`** — one type per block input, same order and length as
        `Nx.block/4` inputs. Before building the custom call, each operand is
        converted (StableHLO `convert`) when its element type differs from the
        requested type; shapes are unchanged. Use this when the native kernel’s
        FFI signature expects dtypes that may differ from the traced expression
        types (for example after promotion rules).
  """

  @enforce_keys [:call_target_name]

  defstruct [
    :call_target_name,
    api_version: 4,
    backend_config: nil,
    attributes: [],
    mlir_attributes: [],
    extra_results: [],
    operand_element_types: :default
  ]

  @type t :: %__MODULE__{
          call_target_name: String.t(),
          api_version: pos_integer(),
          backend_config: String.t() | nil,
          attributes: [{String.t(), String.t()}],
          mlir_attributes: [{String.t(), String.t()}],
          extra_results: [{Nx.Type.t(), tuple()}],
          operand_element_types: :default | [Nx.Type.t()]
        }
end
