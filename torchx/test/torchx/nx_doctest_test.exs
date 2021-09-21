defmodule Torchx.NxDoctestTest do
  @moduledoc """
  Import Nx' doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.BackendTest.

  ## Not callbacks

    * default_backend/1

  ## PyTorch rounds differently than Elixir

    * round/1

  ## Result mismatch in less significant digits (precision issue)

    * logistic/1
    * cos/1
    * erfc/1

  ## Callback or test requires explicit backend transfer to BinaryBackend

    * to_binary/2
    * to_flat_list/2
    * random_uniform/4
    * random_normal/4
    * atan2/2

  ## Not all options are supported

    * argmax/2
    * argmin/2
    * to_batched_list/3

  ## Unsigned ints not supported by Torch

    * quotient/2
    * mean/2

  ## Output mismatch

    * qr/2
    * cholesky/1

  ## Not implemented yet or depends on not implemented

    * window_mean/3
    * window_sum/3
    * map/3
    * norm/2
    * all_close?/3
    * bitcast/2

  """

  # TODO: Move these explanations alongside the functions below
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
        to_binary: 2,
        to_batched_list: 3,
        default_backend: 1,
        round: 1,
        argmax: 2,
        argmin: 2,
        logistic: 1,
        random_uniform: 4,
        random_normal: 4,
        broadcast: 3,
        slice_axis: 5,
        outer: 2,
        qr: 2,
        quotient: 2,
        window_mean: 3,
        mean: 2,
        map: 3,
        to_flat_list: 2,
        cholesky: 1,
        norm: 2,
        all_close?: 3,
        atan2: 2,
        bitcast: 2,
        cbrt: 1,
        cos: 1,
        erfc: 1
      )
      |> Kernel.++([:moduledoc])
end
