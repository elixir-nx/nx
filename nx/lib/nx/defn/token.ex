defmodule Nx.Defn.Token do
  @moduledoc """
  A `defn` token used by hooks.

  ## Documentation for compilers

  The token has a `hooks` field as a list of maps of the shape:

      %{
        expr: Nx.Tensor.t | Nx.Container.t,
        name: atom(),
        callback: (Nx.Tensor.t | Nx.Container.t -> term()) | nil,
        metadata: map()
      }

  The `hooks` field must only be accessed by `defn` compilers.

  The `metadata` field may contain:
    * `:partitions` - list of partition IDs (integers) where the hook should execute.
      If not specified, the hook executes on all partitions.
  """

  # Hooks are stored with the hooks declared first
  # at the end of the list.
  defstruct hooks: []

  @doc false
  def new do
    %Nx.Defn.Token{}
  end

  @doc false
  def add_hook(%Nx.Defn.Token{} = token, expr, name, callback)
      when is_atom(name) and (is_function(callback) or is_nil(callback)) do
    # Extract metadata from the expression if present
    metadata = extract_metadata(expr)
    hook = %{expr: expr, name: name, callback: callback, metadata: metadata}
    update_in(token.hooks, &[hook | &1])
  end

  defp extract_metadata(%Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_expr, metadata]}}),
    do: metadata

  defp extract_metadata(_), do: %{}

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
