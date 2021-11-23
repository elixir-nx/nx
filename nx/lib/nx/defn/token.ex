defmodule Nx.Defn.Token do
  @moduledoc """
  A `defn` token used by hooks.

  ## Documentation for compilers

  The token has a `hooks` field as a list of maps of the shape:

      %{
        expr: Nx.Tensor.t | Nx.Container.t,
        name: atom(),
        callback: (Nx.Tensor.t | Nx.Container.t -> term()) | nil
      }

  The `hooks` field must only be accessed by `defn` compilers.
  """

  defstruct hooks: []

  @doc false
  def new do
    %Nx.Defn.Token{}
  end

  @doc false
  def add_hook(%Nx.Defn.Token{} = token, expr, name, callback)
      when is_atom(name) and (is_function(callback) or is_nil(callback)) do
    hook = %{expr: expr, name: name, callback: callback}
    update_in(token.hooks, &[hook | &1])
  end

  defimpl Nx.Container do
    def traverse(%{hooks: hooks} = token, acc, fun) do
      {hooks, acc} =
        Enum.map_reduce(hooks, acc, fn %{expr: expr} = hook, acc ->
          {expr, acc} = fun.(expr, acc)
          {%{hook | expr: expr}, acc}
        end)

      {%{token | hooks: hooks}, acc}
    end

    def reduce(%{hooks: hooks}, acc, fun) do
      Enum.reduce(hooks, acc, fn %{expr: expr}, acc -> fun.(expr, acc) end)
    end
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(%{hooks: hooks}, opts) do
      concat([
        color("#Nx.Defn.Token<", :map, opts),
        to_doc(Enum.map(hooks, & &1.name), opts),
        color(">", :map, opts)
      ])
    end
  end
end
