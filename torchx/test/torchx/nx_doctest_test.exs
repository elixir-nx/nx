defmodule Torchx.NxDoctestTest do
  @moduledoc """
  Import Nx' doctest and run them on the Torchx backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on Torchx.BackendTest.
  """

  # TODO: Add backend tests for the doctests that are excluded

  use Torchx.Case, async: true

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
    # atan2 - depends on to_binary
    atan2: 2,
    # bitcast - no API available
    bitcast: 2,
    # default_backend - specific to BinaryBackend
    default_backend: 1,
    # to_binary - not supported, must call backend_transfer before
    to_binary: 2,
    # to_flat_list - depends on to_binary
    to_flat_list: 2,
    # random_uniform - depends on to_binary
    random_uniform: 4
  ]

  doctest Nx,
    except:
      Torchx.Backend.__unimplemented__()
      |> Enum.map(fn {fun, arity} -> {fun, arity - 1} end)
      |> Kernel.++(@temporarily_broken_doctests)
      |> Kernel.++(@rounding_error_doctests)
      |> Kernel.++(@inherently_unsupported_doctests)
      |> Kernel.++([:moduledoc])

  describe "rounding error tests" do
    test "atanh/1" do
      assert_tensor(Nx.tensor(0.5493061542510986) == Nx.atanh(Nx.tensor(0.5)))
    end

    test "ceil/1" do
      assert_tensor(Nx.tensor(-0.0) == Nx.ceil(Nx.tensor(-0.5)))
      assert_tensor(Nx.tensor(1.0) == Nx.ceil(Nx.tensor(0.5)))
    end

    test "cos/1" do
      assert_tensor(
        Nx.tensor([-1.0, 0.4999999701976776, -1.0]) ==
          Nx.cos(Nx.tensor([-:math.pi(), :math.pi() / 3, :math.pi()]))
      )
    end

    test "cosh/1" do
      assert_tensor(
        Nx.tensor([11.591955184936523, 1.600286841392517, 11.591955184936523]) ==
          Nx.cosh(Nx.tensor([-:math.pi(), :math.pi() / 3, :math.pi()]))
      )
    end

    test "erfc/1" do
      assert_tensor(
        Nx.tensor([1.0, 0.4795001149177551, 0.0]) == Nx.erfc(Nx.tensor([0, 0.5, 10_000]))
      )
    end

    test "erf_inv/1" do
      assert_tensor(
        Nx.tensor([0.0, 0.4769362807273865, 0.8134198188781738]) ==
          Nx.erf_inv(Nx.tensor([0, 0.5, 0.75]))
      )
    end

    test "round/1" do
      assert_tensor(
        Nx.tensor([-2.0, -0.0, 0.0, 2.0]) == Nx.round(Nx.tensor([-1.5, -0.5, 0.5, 1.5]))
      )
    end

    test "logistic/1" do
      assert_tensor(
        Nx.tensor([0.18242552876472473, 0.622459352016449]) == Nx.logistic(Nx.tensor([-1.5, 0.5]))
      )
    end
  end
end
