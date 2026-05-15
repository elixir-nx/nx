defmodule EXLA.CustomCall.Spec do
  @moduledoc """
  Result of `EXLA.CustomCall.call/4` when lowering a tagged `Nx.block/4` to
  `stablehlo.custom_call`.

    * **`call_target_name`** — XLA FFI handler name (`call_target_name` on the op).

    * **`attributes`** — Optional `{name, attr}` pairs, default `[]`, merged into
      the `backend_config` dictionary on `stablehlo.custom_call` (StableHLO’s name
      for that attribute). Each `name` must be a **BEAM binary** MLIR identifier; each
      `attr` must be a **BEAM binary** with valid MLIR attribute syntax for the RHS after
      `name = ` (for example `{"k", "42 : i64"}`). An empty list omits the dictionary
      from the op.

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

  defstruct [:call_target_name, attributes: [], operand_element_types: :default]

  @type t :: %__MODULE__{
          call_target_name: String.t(),
          attributes: [{String.t(), String.t()}],
          operand_element_types: :default | [Nx.Type.t()]
        }
end
