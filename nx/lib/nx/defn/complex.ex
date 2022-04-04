defmodule Nx.Defn.Complex do
  def new(re, im \\ 0) do
    re
    |> Complex.new(im)
    |> Nx.tensor()
  end
end
