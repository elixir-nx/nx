defmodule Nx.Names do
  @moduledoc false

  # Collection of utilities for working on named tensors

  @doc false
  def validate!(names, shape) do
    # Every dimension must be named, or none
    # of the dimensions can be named, `nil` and
    # atoms are the only valid names
    n_dims = tuple_size(shape)

    if names do
      n_names = length(names)

      if n_names != n_dims do
        raise ArgumentError,
              "invalid names for tensor of rank #{n_dims}," <>
                " when specifying names every dimension must" <>
                " have a name or be nil"
      else
        names
      end
    else
      names = for _ <- 1..n_dims, do: nil
    end
  end
end
