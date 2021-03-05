defmodule Torchx.NxDoctestTest do
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
        default_backend: 2,
        add: 2,
        round: 1,
        norm_nuclear: 1,
        outer: 2,
        random_uniform: 4,
        dot: 2,
        random_normal: 4,
        qr: 2,
        quotient: 2,
        window_mean: 3,
        iota: 2,
        transpose: 2,
        mean: 2,
        dot: 4,
        left_shift: 2,
        map: 3,
        backend_transfer: 1,
        to_flat_list: 2,
        right_shift: 2,
        norm_inf: 3,
        cholesky: 1,
        norm: 2,
        norm_integer: 3,
        all_close?: 3,
        atan2: 2,
        lu: 2,
        sum: 2
      )
    |> Kernel.++([:moduledoc])
end
