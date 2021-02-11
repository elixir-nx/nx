defmodule Nx.GradHelpers do
  @doc """
  Checks the gradient of numerical function `func`.

  You must hold the function constant on every other
  variable with a partial application of `func`.
  """
  def check_grads!(func, grad_func, x, eps \\ 1.0e-4) do
    est_grad = finite_differences(func, x, eps)
    comp_grad = grad_func.(x)
    approx_equal?(est_grad, comp_grad, x, eps)
  end

  defp approx_equal?(lhs, rhs, x, eps) do
    [value] = Nx.to_flat_list(Nx.abs(Nx.subtract(lhs, rhs)))

    unless value < eps do
      raise """
      expected

      #{inspect(lhs)}

      to be #{eps} within

      #{inspect(rhs)}

      for input

      #{inspect(x)}
      """
    end
  end

  defp finite_differences(func, x, eps) do
    Nx.divide(
      Nx.subtract(
        func.(Nx.add(x, Nx.divide(eps, 2.0))),
        func.(Nx.subtract(x, Nx.divide(eps, 2.0)))
      ),
      eps
    )
  end
end

defmodule Nx.ProcessDevice do
  @behaviour Nx.Device

  def allocate(data, _type, _shape, opts) do
    key = Keyword.fetch!(opts, :key)
    Process.put(key, data)
    {__MODULE__, key}
  end

  def read(key), do: Process.get(key) || raise("deallocated")

  def deallocate(key), do: if(Process.delete(key), do: :ok, else: :already_deallocated)
end

ExUnit.start()
