defmodule EXLA.Backend do
  @moduledoc """
  EXLA Nx backend implementation for usage of EXLA outside of defn.

  The EXLA backend is designed for rapid prototyping of accelerated Nx programs
  and ensures consistency from prototype to implementation. Generally, you'll want
  to experiment with ideas before organizing them into a project. Backends allow
  you to experiment with Nx in "eager mode" before utilizing the stricter, but
  more performant constructs such as `defn` and JIT compilation.

  While you can also prototype with the BinaryBackend, the EXLA backend offers
  the following advantages:

    1) Performance - The EXLA backend executes functions on CPU/GPU/TPU.

    2) Consistency - You may encounter slight inconsistencies in behavior between
    the BinaryBackend and the EXLA backend which leads to different behavior when
    changing "eager mode" code to compiled code. Using EXLA in "eager mode" ensures
    behavior is consistent when transitioning from eager mode to compiled.
  """

  @behaviour Nx.Backend

  @unsupported_functions [
    scalar: 3,
    from_binary: 3,
    to_batched_list: 3
  ]

  @creation_functions [
    iota: 3,
    random_uniform: 4,
    random_normal: 4,
    eye: 2
  ]

  @supported_functions Nx.Backend.behaviour_info(:callbacks)
                        -- @unsupported_functions
                        -- @creation_functions

  @impl true
  def eye(%{shape: shape, type: type, names: names}, _) do
    fun = fn ->
      Nx.eye(shape, type: type, names: names, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def iota(%{shape: shape, type: type, names: names}, axis, _) do
    fun = fn ->
      Nx.iota(shape, type: type, names: names, axis: axis, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def random_uniform(%{shape: shape, type: type, names: names}, min, max, _) do
    fun = fn ->
      Nx.random_uniform(shape, min, max, type: type, names: names, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def random_normal(%{shape: shape, type: type, names: names}, mu, sigma, _) do
    fun = fn ->
      Nx.random_normal(shape, mu, sigma, type: type, names: names, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  for {fun, arity} <- @unsupported_functions do
    arguments = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    defdelegate unquote(fun)(unquote_splicing(arguments)), to: Nx.BinaryBackend
  end

  for {fun, arity} <- @supported_functions do
    arguments = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(arguments)) do
      EXLA.jit(Function.capture(Nx, unquote(fun), unquote(arity)), unquote(arguments))
    end
  end
end