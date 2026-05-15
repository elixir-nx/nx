defprotocol EXLA.CustomCall do
  @moduledoc """
  Extension point for lowering selected `Nx.block/4` blocks to **XLA custom calls**
  (`stablehlo.custom_call` in StableHLO MLIR).

  Other blocks (for example gather-based `take` or FFT) are lowered inline in
  `EXLA.Defn` and do not use this protocol.

  ## When `EXLA.Defn` calls it

  During compilation with `compiler: EXLA`, each `Nx.block(block, inputs, outputs, fn ... end)`
  is processed by this protocol. `EXLA` invokes `call/4` once per block.

  If `call/4` returns `:skip`, EXLA compiles the block's **default callback**
  (the anonymous function body) instead of emitting a custom call.
  Default lowerings are provided for `Nx.Block.LinAlg.QR` and `Nx.Block.LinAlg.Eigh`.

  ## `call/4` arguments

  Callback arity is `call(struct, out, args, client)`, matching
  `Nx.block(block, inputs, outputs, fn ... end)` (block, outputs, inputs, then client).

    * `struct` — the **block** passed as the first argument to `Nx.block/4`
      (your own `defstruct` or an existing block such as `%Nx.Block.LinAlg.QR{}`).

    * `out` — the **output template** tuple passed to `Nx.block/4` (expression
      metadata for shapes and types, not runtime tensors).

    * `args` — list of **input templates**, in the same order as `inputs` in
      `Nx.block/4`.

    * `client` — the active `EXLA.Client` (use e.g. `client.platform` to gate
      host-only lowerings).

  ## `call/4` return value

    * **`:skip`** — this implementation does not apply (unsupported type,
      non-host platform, wrong arity, etc.). The default block implementation
      is used instead.

    * **`{:ok, %EXLA.CustomCall.Spec{}}`** — emit a StableHLO custom call; see
      `EXLA.CustomCall.Spec` for `call_target_name`, optional `attributes`
      (`[{name, attr}]` string pairs for the `stablehlo.custom_call` `backend_config` dictionary), and optional
      `operand_element_types` (operand converts when they differ
      from the lowered inputs).

  ## Dispatch

  The protocol uses `@fallback_to_any true`. Built-in lowerings for known blocks
  live in `defimpl EXLA.CustomCall, for: Any`. Your application or dependency can
  add `defimpl EXLA.CustomCall, for: YourStruct`; that implementation is chosen
  whenever the block is `%YourStruct{}`, instead of the `Any` fallback.

  ## Native handlers

  Emitting a custom call in MLIR is only half of the story: the **target name**
  must be registered with XLA on the relevant platform (typically via a native
  library loaded into the process). That registration is **not** configured
  through `config :exla, ...`; you load or link the native code by the same
  means you would for any other NIF-backed extension.

  ## Example

      defmodule MyApp.CustomQrBlock do
        defstruct []
      end

      defimpl EXLA.CustomCall, for: MyApp.CustomQrBlock do
        def call(_block, {%{type: {kind, size}}, _r_expr}, [_input], %{platform: :host})
            when kind != :c and kind in [:f, :bf] and size in [16, 32, 64] do
          {:ok, %EXLA.CustomCall.Spec{call_target_name: "my_custom_qr_target"}}
        end

        def call(_, _, _, _), do: :skip
      end

  Then use `Nx.block(%MyApp.CustomQrBlock{}, ...)` inside a `defn` compiled with
  `compiler: EXLA`.
  """

  @fallback_to_any true

  @doc """
  Returns `:skip` or `{:ok, %EXLA.CustomCall.Spec{}}`.

  Invoked as `call(struct, out, args, client)`.
  """
  def call(struct, out, args, client)
end

# Default EXLA lowerings for **C-backed custom_call** `Nx.block/4` blocks live
# in this `defimpl ..., for: Any` module. With `@fallback_to_any true` on the
# protocol, applications and libraries can define their own
# `defimpl EXLA.CustomCall, for: SomeStruct` — protocol dispatch uses that
# implementation instead of this fallback when the block matches (you can
# also target a built-in struct such as `Nx.Block...` from your app if needed).
#
defimpl EXLA.CustomCall, for: Any do
  @moduledoc false

  alias EXLA.CustomCall.Spec

  def call(
        %Nx.Block.LinAlg.QR{},
        {%{type: q_type}, _r_expr},
        [%{type: in_type} | _],
        %{platform: :host}
      )
      when elem(q_type, 0) != :c and elem(in_type, 0) != :c do
    qr_cpu_custom_call(in_type)
  end

  # Native target names depend only on the input dtype; output templates may use
  # different element types (e.g. promotion) and must not change the call target.
  def call(%Nx.Block.LinAlg.Eigh{}, _out, [%{type: in_type} | _], %{platform: :host})
      when elem(in_type, 0) != :c do
    eigh_cpu_custom_call(in_type)
  end

  def call(_, _, _, _), do: :skip

  defp qr_cpu_custom_call({kind, _bits}) when kind in [:s, :u] do
    {:ok,
     %Spec{
       call_target_name: "qr_cpu_custom_call_f32",
       operand_element_types: [{:f, 32}]
     }}
  end

  defp qr_cpu_custom_call(in_type) do
    case in_type do
      {:f, 32} -> {:ok, %Spec{call_target_name: "qr_cpu_custom_call_f32"}}
      {:f, 64} -> {:ok, %Spec{call_target_name: "qr_cpu_custom_call_f64"}}
      {:f, 16} -> {:ok, %Spec{call_target_name: "qr_cpu_custom_call_f16"}}
      {:bf, 16} -> {:ok, %Spec{call_target_name: "qr_cpu_custom_call_bf16"}}
      _ -> :skip
    end
  end

  defp eigh_cpu_custom_call({kind, _bits}) when kind in [:s, :u] do
    {:ok,
     %Spec{
       call_target_name: "eigh_cpu_custom_call_f32",
       operand_element_types: [{:f, 32}]
     }}
  end

  defp eigh_cpu_custom_call(in_type) do
    case in_type do
      {:f, 32} -> {:ok, %Spec{call_target_name: "eigh_cpu_custom_call_f32"}}
      {:f, 64} -> {:ok, %Spec{call_target_name: "eigh_cpu_custom_call_f64"}}
      _ -> :skip
    end
  end
end
