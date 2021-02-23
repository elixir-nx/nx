defmodule Nx.GradHelpers do
  @doc """
  Checks the gradient of numerical function `func`.

  You must hold the function constant on every other
  variable with a partial application of `func`.
  """
  def check_grads!(func, grad_func, x, opts \\ []) when is_list(opts) do
    eps = opts[:eps] || 1.0e-4
    step = opts[:step] || 1.0e-4
    est_grad = finite_differences(func, x, step)
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

  defp finite_differences(func, x, step) do
    Nx.divide(
      Nx.subtract(
        func.(Nx.add(x, Nx.divide(step, 2.0))),
        func.(Nx.subtract(x, Nx.divide(step, 2.0)))
      ),
      step
    )
  end
end

ExUnit.start()
