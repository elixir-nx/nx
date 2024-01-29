defmodule Torchx.NxLinAlgDoctestTest do
  @moduledoc """
  Import Nx.LinAlg's doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.NxTest.
  """

  use ExUnit.Case, async: true

  # Results match but there are arounding errors
  @rounding_error_doctests [
    cholesky: 1,
    matrix_power: 2,
    qr: 2,
    triangular_solve: 3,
    solve: 2,
    invert: 1,
    determinant: 1,
    pinv: 2,
    least_squares: 3
  ]

  # Results do not match but properties are respected
  # All of these have tests in nx_linalg_test.exs
  @property_doctests [
    eigh: 2,
    lu: 2,
    svd: 2
  ]

  @type_incompatibility_doctests [
    # u64 is not supported by Torchx
    norm: 2
  ]

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  doctest Nx.LinAlg,
    except:
      @type_incompatibility_doctests
      |> Kernel.++(@rounding_error_doctests)
      |> Kernel.++(@property_doctests)
end
