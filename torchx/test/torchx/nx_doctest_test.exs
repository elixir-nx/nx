defmodule Torchx.NxDoctestTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  doctest Nx,
    except:
      Torchx.Backend.__unimplemented__() |> Enum.map(fn {fun, arity} -> {fun, arity - 1} end)
end
