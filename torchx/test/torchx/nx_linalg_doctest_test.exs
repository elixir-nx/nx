defmodule Torchx.NxLinAlgDoctestTest do
  @moduledoc """
  Import Nx.LinAlg's doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.BackendTest.
  """

  # TODO: Add backend tests for the doctests that are excluded

  use ExUnit.Case, async: true

  @rounding_error_doctests [
    triangular_solve: 3,
    # The expected result for some tests isn't the same
    # even though the A = P.L.U property is maintained for lu/2
    lu: 2
  ]

  @temporarily_broken_doctests [
    # norm - reduce_max not implemented
    norm: 2,
    # qr - Torchx: "geqrf_cpu" not implemented for 'Long' in NIF.qr/2
    qr: 2,
    # invert - Depends on QR
    invert: 1,
    # solve - Depends on QR
    solve: 2,
    # cholesky - returns the transposed result
    cholesky: 1
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
end
