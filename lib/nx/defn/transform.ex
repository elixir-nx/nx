defmodule Nx.Defn.Transform do
  @moduledoc """
  Defines the behaviour for `defn` transforms.

  A transform is given by calling `Nx.Defn.Kernel.transform/3`.
  """

  @doc """
  The transform callback invoked for the transformation.

  It receives the `env`, the variable `counter`, the `meta` for
  the `transform` call, the `ast`, and a keyword list of options.

  It must return an upgraded `counter` if any new variable is
  created and the updated AST.
  """
  @callback __transform__(
              env :: Macro.Env.t(),
              counter :: integer,
              meta :: keyword,
              ast :: Macro.t(),
              opts :: keyword
            ) :: {counter :: integer, ast :: Macro.t()}
end
