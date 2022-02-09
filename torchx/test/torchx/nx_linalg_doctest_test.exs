defmodule Torchx.NxLinAlgDoctestTest do
  @moduledoc """
  Import Nx.LinAlg's doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.NxTest.
  """

  # TODO: Add backend tests for the doctests that are excluded

  use ExUnit.Case, async: true

  # Results match but there are arounding errors
  @rounding_error_doctests [
    cholesky: 1,
    matrix_power: 2,
    qr: 2,
    triangular_solve: 3
  ]

  # Results do not match but properties are respected
  @property_doctests [
    eigh: 2,
    lu: 2,
    svd: 2
  ]

  @temporarily_broken_doctests [
    # unsigned 64 bit integer support
    determinant: 1,
    norm: 2,
  ]

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  doctest Nx.LinAlg,
    except:
      Torchx.Backend.__unimplemented__()
      |> Enum.map(fn {fun, arity} -> {fun, arity - 1} end)
      |> Kernel.++(@temporarily_broken_doctests)
      |> Kernel.++(@rounding_error_doctests)
      |> Kernel.++(@property_doctests)
end
