defmodule Nx.Helpers do
  import Nx.Testing

  @doc """
  Checks the gradient of numerical function `func`.

  You must hold the function constant on every other
  variable with a partial application of `func`.
  """
  def check_grads!(func, grad_func, x, opts)
      when is_function(func) and is_function(grad_func) and is_list(opts) do
    atol = opts[:atol] || 1.0e-7
    rtol = opts[:rtol] || 1.0e-4
    step = opts[:step] || 1.0e-4

    est_grad = finite_differences(func, x, step)
    comp_grad = grad_func.(x)
    assert_all_close(comp_grad, est_grad, atol: atol, rtol: rtol, input: x)
  end

  def check_grads!(func, grad_func, x) when is_function(func) and is_function(grad_func) do
    check_grads!(func, grad_func, x, [])
  end

  def check_grads!(func, x, opts) when is_function(func) and is_list(opts) do
    grad = fn x -> x |> Nx.Defn.grad(func) |> Nx.sum() end
    check_grads!(func, grad, x, opts)
  end

  def check_grads!(func, x) when is_function(func), do: check_grads!(func, x, [])

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
