defmodule EXLA.CustomCall.Spec do
  @moduledoc """
  Result of `EXLA.CustomCall.call/4` when lowering a tagged `Nx.block/4` to
  `stablehlo.custom_call`.

    * **`call_target_name`** — XLA FFI handler name (`call_target_name` on the op).

    * **`backend_config`** — Optional StableHLO dictionary attribute (`nil` omits it).
      Same constraints as `EXLA.MLIR.Value.custom_call/4`.

    * **`operand_element_types`** — How operand SSA values are presented to the handler:

      * **`:infer`** (default) — use each lowered operand’s element type as produced
        from the block inputs. No extra converts.

      * **`[Nx.Type.t(), ...]`** — one type per block input, same order and length as
        `Nx.block/4` inputs. Before building the custom call, each operand is
        converted (StableHLO `convert`) when its element type differs from the
        requested type; shapes are unchanged. Use this when the native kernel’s
        FFI signature expects dtypes that may differ from the traced expression
        types (for example after promotion rules).
  """

  @enforce_keys [:call_target_name]

  defstruct [:call_target_name, backend_config: nil, operand_element_types: :infer]

  @type t :: %__MODULE__{
          call_target_name: String.t(),
          backend_config: map() | nil,
          operand_element_types: :infer | [Nx.Type.t()]
        }
end
