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
    # all_close? - depends on all? which is not implemented
    all_close?: 3,
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

  @inherently_unsupported_doctests [
    # atanh - rounding error
    atanh: 1,
    # atan2 - depends on to_binary
    atan2: 2,
    # bitcast - no API available
    bitcast: 2,
    # ceil - sign error -0.0 vs 0.0
    ceil: 1,
    # cos - rounding error
    cos: 1,
    # cosh - rounding error
    cosh: 1,
    # default_backend - specific to BinaryBackend
    default_backend: 1,
    # erfc - rounding error (on some archs)
    erfc: 1,
    # erf_inv - rounding error (on some archs)
    erf_inv: 1,
    # round - rounds to different direction
    round: 1,
    # to_binary - not supported, must call backend_transfer before
    to_binary: 2,
    # to_flat_list - depends on to_binary
    to_flat_list: 2,
    # logistic - rounding error
    logistic: 1,
    # random_uniform - depends on to_binary
    random_uniform: 4
  ]

  doctest Nx,
    except:
      Torchx.Backend.__unimplemented__()
      |> Enum.map(fn {fun, arity} -> {fun, arity - 1} end)
      |> Kernel.++(@temporarily_broken_doctests)
      |> Kernel.++(@inherently_unsupported_doctests)
      |> Kernel.++([:moduledoc])
end
