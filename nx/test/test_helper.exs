defmodule Nx.GradHelpers do
  @doc """
  Checks the gradient of numerical function `func`.

  You must hold the function constant on every other
  variable with a partial application of `func`.
  """
  def check_grads!(func, grad_func, x, opts \\ []) when is_list(opts) do
    atol = opts[:atol] || 1.0e-7
    rtol = opts[:rtol] || 1.0e-4
    step = opts[:step] || 1.0e-4
    est_grad = finite_differences(func, x, step)
    comp_grad = grad_func.(x)
    approx_equal?(comp_grad, est_grad, x, atol, rtol)
  end

  defp approx_equal?(lhs, rhs, x, atol, rtol) do
    unless Nx.all_close?(lhs, rhs, atol: atol, rtol: rtol) == Nx.tensor(1, type: {:u, 8}) do
      raise """
      expected

      #{inspect(lhs)}

      to be within tolerance of

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
