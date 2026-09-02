defmodule EXLA.Defn.CallbackError do
  @moduledoc false
  defexception [:kind, :reason, :stacktrace]

  @impl true
  def message(%{kind: kind, reason: reason, stacktrace: stacktrace}) do
    Exception.format_banner(kind, reason, stacktrace)
  end
end
