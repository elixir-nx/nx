defmodule Formulas do
  for {name, {_, _, formula}} <- Nx.Shared.unary_math_funs() do
    @doc """
    #{formula}
    """
    def unquote(name)(z), do: z
  end
end
