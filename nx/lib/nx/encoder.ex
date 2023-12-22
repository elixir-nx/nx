defmodule Nx.Encoder do
  @moduledoc """
  Set of utilities to encode features.
  """

  import Nx.Defn, only: [defn: 2]

  # TO-DO
  defn one_hot_encoder(t) do
    tensor = Nx.to_tensor(t)
  end
end
