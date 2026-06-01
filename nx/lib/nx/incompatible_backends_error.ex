defmodule Nx.Defn.IncompatibleBackendsError do
  @moduledoc """
  Exception raised when two tensors with incompatible backend implementations
  are used together in a single Nx operation.
  """

  defexception [:message, :backend1, :backend2]

  @impl true
  def exception(opts) do
    backend1 = Keyword.fetch!(opts, :backend1)
    backend2 = Keyword.fetch!(opts, :backend2)

    detail =
      if backend1 == Nx.Defn.Expr or backend2 == Nx.Defn.Expr do
        "This may mean you are passing a tensor to defn/jit as an optional argument " <>
          "or as closure in an anonymous function. For efficiency, it is preferred " <>
          "to always pass tensors as required arguments instead. Alternatively, you " <>
          "could call Nx.backend_copy/1 on the tensor, however this will copy its " <>
          "value and inline it inside the defn expression"
      else
        "You may need to call Nx.backend_transfer/2 (or Nx.backend_copy/2) " <>
          "on one or both of them to transfer them to a common implementation"
      end

    message =
      "cannot invoke Nx function because it relies on two incompatible tensor implementations: " <>
        "#{inspect(backend1)} and #{inspect(backend2)}. " <> detail

    %__MODULE__{message: message, backend1: backend1, backend2: backend2}
  end
end
