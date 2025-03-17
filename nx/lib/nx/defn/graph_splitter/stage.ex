defmodule Nx.Defn.GraphSplitter.Stage do
  @typedoc """
  A stage in the graph splitter.

  `:arguments` is a map of the id of the corresponding Nx.Defn.Expr :parameter
  node to the source {stage_id, output_container_position} of the argument
  and the index of the argument in the current stage.
  """
  @type t :: %__MODULE__{
          id: reference(),
          expr: %{__struct__: Nx.Defn.Expr},
          arguments: [%{source: {reference() | nil, non_neg_integer()}}]
        }

  defstruct [:id, :expr, :arguments]
end
