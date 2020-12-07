defmodule Nx.Grad do
  # @behaviour Nx.Defn.Transform

  def __transform__(_env, version, _meta, {_var, ast}, _opts) do
    {version, transform(ast)}
  end

  defp transform({{:., dot_meta, [Nx, name]}, meta, args}) do
    {{:., dot_meta, [Nx.Grad, name]}, meta, args}
  end

  import Nx.Defn

  @doc """
  The derivative of `Nx.tanh/2`.
  """
  defn tanh(t), do: 1.0 - Nx.power(Nx.tanh(t), 2)
end
