defmodule Torchx.NxDoctestTest do
  @moduledoc """
  # Why some doctests are excluded?

  ## Not callbacks

  * default_backend/1

  ## Python rounds differently, than Elixir (for now)

  * round/1

  ## Result mismatch in less significant digits (precision issue)

  * logistic/1

  ## Callback or test requires explicit backend transfer to BinaryBackend

  * to_binary/2
  * to_flat_list/2
  * random_uniform/4
  * random_normal/4
  * atan2/2

  ## Not all options are supported

  * argmax/2
  * argmin/2

  ## Unsigned ints not supported by Torch

  * quotient/2
  * mean/2

  ## Output mismatch

  * qr/2
  * dot/4
  * cholesky/1

  ## Not all data types supported

  * dot/2

  ## Not implemented yet or depends on not implemented

  * window_mean/3
  * window_sum/3
  * map/3
  * norm/2
  * all_close?/3
  * bitcast/2

  ## WIP

  * transpose/2

  """

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
        transpose: 2,
        mean: 2,
        dot: 4,
        map: 3,
        to_flat_list: 2,
        cholesky: 1,
        norm: 2,
        all_close?: 3,
        atan2: 2,
        bitcast: 2
      )
      |> Kernel.++([:moduledoc])
end
