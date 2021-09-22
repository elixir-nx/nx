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

  doctest Nx,
    except:
      Torchx.Backend.__unimplemented__()
      |> Enum.map(fn {fun, arity} -> {fun, arity - 1} end)
      |> Kernel.++(
        # to_binary - not supported, must call backend_transfer before
        to_binary: 2,
        # to_batched_list - Shape mismatch due to unsupported options in some tests
        to_batched_list: 3,
        # default_backend - Test expectes BinaryBackend, but we return Torchx.Backend
        default_backend: 1,
        # round - rounds to different direction
        round: 1,
        # argmax - tie_break option not supported
        argmax: 2,
        # argmin - tie_break option not supported
        argmin: 2,
        # logistic - rounding error
        logistic: 1,
        # random_uniform - depends on to_binary
        random_uniform: 4,
        # broadcast - shape mismatch
        broadcast: 3,
        # slice_axis - expects scalar starts and receives tensors
        slice_axis: 5,
        # outer - shape mismatch
        outer: 2,
        # quotient - Torchx expects a input tensor but receives a number as input
        quotient: 2,
        # window_mean - depends on window_sum
        window_mean: 3,
        # mean - Torchx expects a input tensor but receives a number as input
        mean: 2,
        # map - operation not supported
        map: 3,
        # to_flat_list - depends on to_binary
        to_flat_list: 2,
        # all_close? - Depends on all? which is not supported
        all_close?: 3,
        # atan2 - depends on to_binary
        atan2: 2,
        # bitcast - not supported
        bitcast: 2,
        # cbrt - not supported
        cbrt: 1,
        # cos - removed due to rounding error
        cos: 1,
        # atanh - removed due to rounding error
        atanh: 1,
        # concatenate - unsupported on torchx
        concatenate: 3,
        # Stack - also uses concatenate internally
        stack: 2,
        # ceil - sign error -0.0 vs 0.0
        ceil: 1,
        # cosh - removed due to rounding error
        cosh: 1,
        # erf_inv - removed due to rounding error
        erf_inv: 1,
        # dot - Batching not supported
        dot: 6,
        # slice - expects numerical start indices, but now receives tensors,
        slice: 4
      )
      |> Kernel.++([:moduledoc])
end
