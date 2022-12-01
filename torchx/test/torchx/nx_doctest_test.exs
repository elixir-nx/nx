defmodule Torchx.NxDoctestTest do
  @moduledoc """
  Import Nx' doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.NxTest.
  """

  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  @rounding_error_doctests [
    atanh: 1,
    ceil: 1,
    cos: 1,
    cosh: 1,
    erfc: 1,
    erf_inv: 1,
    round: 1,
    sigmoid: 1,
    fft: 2,
    ifft: 2
  ]

  if Application.get_env(:torchx, :is_apple_arm64) do
    @os_rounding_error_doctests [sin: 1]
  else
    case :os.type() do
      {:win32, _} -> @os_rounding_error_doctests [expm1: 1, erf: 1]
      _ -> @os_rounding_error_doctests []
    end
  end

  @unrelated_doctests [
    default_backend: 1
  ]

  @inherently_unsupported_doctests [
    # as_type - the rules change per type
    as_type: 2,
    # no API available - bit based
    count_leading_zeros: 1,
    population_count: 1,
    # no API available - function based
    map: 3,
    window_reduce: 5,
    reduce: 4
  ]

  doctest Nx,
    except:
      @rounding_error_doctests
      |> Kernel.++(@os_rounding_error_doctests)
      |> Kernel.++(@inherently_unsupported_doctests)
      |> Kernel.++(@unrelated_doctests)
      |> Kernel.++([:moduledoc])
end
