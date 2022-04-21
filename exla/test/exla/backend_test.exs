defmodule EXLA.BackendTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  doctest Nx, except: [compatible?: 2, default_backend: 1]
end
