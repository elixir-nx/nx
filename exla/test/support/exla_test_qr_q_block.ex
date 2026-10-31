# Test-only block tag + `EXLA.CustomCall` impl whose output is only the Q of a QR. The
# `qr_cpu_custom_call_f32` target also returns R, which the spec declares in `extra_results`.
defmodule EXLA.Test.QRQBlock do
  @moduledoc false
  defstruct []
end

defimpl EXLA.CustomCall, for: EXLA.Test.QRQBlock do
  def call(_, %{type: {:f, 32}}, [%{type: {:f, 32}, shape: {m, n}}], %{platform: :host}) do
    {:ok,
     %EXLA.CustomCall.Spec{
       call_target_name: "qr_cpu_custom_call_f32",
       extra_results: [{{:f, 32}, {min(m, n), n}}]
     }}
  end

  def call(_, _, _, _), do: :skip
end
