defmodule EXLA.Defn.CallbackError do
  @moduledoc false
  defexception [:kind, :reason, :stacktrace]

  @impl true
  def message(%{kind: kind, reason: reason}) do
    Exception.format_banner(kind, reason, [])
  end
end
