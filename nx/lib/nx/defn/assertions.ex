defmodule Nx.Defn.Assertions do
  @moduledoc false
  # Holds assertions for transforms
  import Nx.Defn, only: [deftransform: 2]

  deftransform __assert_shape_pattern__(tensor, assertion_fun) do
    assertion_fun.(tensor)
  end
end
