defmodule Nx.Helpers do
  import ExUnit.Assertions

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
    assert_all_close(comp_grad, est_grad, x, atol, rtol)
  end

  @doc """
  Asserts `lhs` is close to `rhs`.
  """
  def assert_all_close(lhs, rhs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    unless Nx.all_close(lhs, rhs, atol: atol, rtol: rtol, equal_nan: opts[:equal_nan]) ==
             Nx.tensor(1, type: {:u, 8}) do
      flunk("""
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}
      """)
    end
  end

  defp assert_all_close(lhs, rhs, x, atol, rtol) do
    unless Nx.all_close(lhs, rhs, atol: atol, rtol: rtol) == Nx.tensor(1, type: {:u, 8}) do
      flunk("""
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}

      for input

      #{inspect(x)}
      """)
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

  @doc """
  Compares `Nx.Defn.grad/2` of a scalar loss with a central difference
  on each element.

  Vectorized axes are summed into that scalar. The partial of each element
  still matches when those entries do not depend on each other.
  """
  def check_scalar_grad!(tensor, fun, opts \\ []) when is_list(opts) do
    step = opts[:step] || 1.0e-5
    atol = opts[:atol] || 1.0e-6
    rtol = opts[:rtol] || 1.0e-6

    loss = fn t -> t |> fun.() |> Nx.devectorize() |> Nx.sum() end
    analytic = Nx.Defn.grad(tensor, loss)
    numeric = central_gradient(loss, tensor, step)
    Nx.Testing.assert_all_close(analytic, numeric, atol: atol, rtol: rtol)
  end

  defp central_gradient(loss, tensor, step) do
    devectorized = Nx.devectorize(tensor)
    values = Nx.to_flat_list(devectorized)
    type = tensor.type
    axes = tensor.vectorized_axes

    values
    |> Enum.with_index(fn value, index ->
      plus = replace_flat(values, index, value + step, devectorized.shape, type, axes)
      minus = replace_flat(values, index, value - step, devectorized.shape, type, axes)
      (Nx.to_number(loss.(plus)) - Nx.to_number(loss.(minus))) / (2 * step)
    end)
    |> Nx.tensor(type: type)
    |> Nx.reshape(devectorized.shape)
    |> restore_vectorized(axes)
  end

  defp replace_flat(values, index, new_value, shape, type, axes) do
    values
    |> List.replace_at(index, new_value)
    |> Nx.tensor(type: type)
    |> Nx.reshape(shape)
    |> restore_vectorized(axes)
  end

  defp restore_vectorized(tensor, []), do: tensor
  defp restore_vectorized(tensor, axes), do: Nx.vectorize(tensor, axes)
end
