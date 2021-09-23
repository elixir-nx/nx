defmodule Torchx.NxDoctestTest do
  @moduledoc """
  Import Nx' doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.BackendTest.
  """

  # TODO: Add backend tests for the doctests that are excluded

  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  @temporarily_broken_doctests [
    # argmax - tie_break option not supported
    argmax: 2,
    # argmin - tie_break option not supported
    argmin: 2,
    # broadcast - shape mismatch in one test
    broadcast: 3,
    # dot - Batching not supported
    dot: 6,
    # mean - Torchx expects a input tensor but receives a number as input
    mean: 2,
    # outer - shape mismatch in some tests
    outer: 2,
    # quotient - Torchx expects a input tensor but receives a number as input
    quotient: 2,
    # slice - expects numerical start indices, but now receives tensors,
    slice: 4,
    # slice_axis - expects scalar starts and receives tensors
    slice_axis: 5,
    # stack - fails in some tests
    stack: 2,
    # to_batched_list - Shape mismatch due to unsupported options in some tests
    to_batched_list: 3,
    # window_mean - depends on window_sum which is not implemented
    window_mean: 3
  ]

  @rounding_error_doctests [
    atanh: 1,
    ceil: 1,
    cos: 1,
    cosh: 1,
    erfc: 1,
    erf_inv: 1,
    round: 1,
    logistic: 1
  ]

  @inherently_unsupported_doctests [
    # bitcast - no API available
    bitcast: 2,
    # default_backend - specific to BinaryBackend
    default_backend: 1
  ]

  doctest Nx,
    except:
      Torchx.Backend.__unimplemented__()
      |> Enum.map(fn {fun, arity} -> {fun, arity - 1} end)
      |> Kernel.++(@temporarily_broken_doctests)
      |> Kernel.++(@rounding_error_doctests)
      |> Kernel.++(@inherently_unsupported_doctests)
      |> Kernel.++([:moduledoc])
end
