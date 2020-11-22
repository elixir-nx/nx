defmodule Nx.Defn.Kernel do
  @moduledoc """
  The API available inside `defn` blocks.

  Many of the macros in this module are conveniences
  that delegate either to `Nx` or `Kernel` namespaces.
  """

  @doc """
  A `+` operator which delegates to `Nx.add/2`.
  """
  defmacro a + b do
    quote do: Nx.add(unquote(a), unquote(b))
  end

  @doc """
  A `/` operator which delegates to `Nx.divide/2`.
  """
  defmacro a / b do
    quote do: Nx.divide(unquote(a), unquote(b))
  end

  @doc """
  A `|>` operator which delegates to `Kernel.|>/2`.
  """
  defmacro left |> right do
    quote do: Kernel.|>(unquote(left), unquote(right))
  end
end
