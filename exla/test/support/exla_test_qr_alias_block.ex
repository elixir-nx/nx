# Test-only block tag + `EXLA.CustomCall` impl used to emit a StableHLO custom_call
# with `call_target_name` `qr_cpu_custom_call_f32_exla_alias` (registered by
# `priv/test/exla_qr_alias.so` when built with `MIX_ENV=test`).
defmodule EXLA.Test.QRAliasBlock do
  @moduledoc false
  defstruct []
end

defimpl EXLA.CustomCall, for: EXLA.Test.QRAliasBlock do
  def call(_, {%{type: {q_kind, q_size}}, _r_expr}, [_tensor], client)
      when q_kind != :c and q_size == 32 and client.platform == :host do
    {:ok, %EXLA.CustomCall.Spec{call_target_name: "qr_cpu_custom_call_f32_exla_alias"}}
  end

  def call(_, _, _, _), do: :skip
end
