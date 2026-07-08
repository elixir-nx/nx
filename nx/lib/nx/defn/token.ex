defmodule Nx.Defn.Token do
  @moduledoc """
  A `defn` token used by hooks.

  Deprecated. Use `hook/2` or `hook/3` instead. Will be removed in a future release.

  ## Documentation for compilers

  The token has a `hooks` field as a list of maps of the shape:

      %{
        expr: Nx.Tensor.t | Nx.Container.t,
        name: atom(),
        callback: (Nx.Tensor.t | Nx.Container.t -> term()) | nil
      }

  The `hooks` field must only be accessed by `defn` compilers.
  """

  # Hooks are stored with the hooks declared first
  # at the end of the list.
  defstruct hooks: []

  @doc false
  def new do
    %__MODULE__{}
  end

  @doc false
  def add_hook(%__MODULE__{} = token, expr, name, callback)
      when is_atom(name) and (is_function(callback) or is_nil(callback)) do
    hook = %{expr: expr, name: name, callback: callback}
    update_in(token.hooks, &[hook | &1])
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
