defmodule Nx.Defn do
  @moduledoc """
  Compile-time numerical definitions.

  A numerical function is a subset of Elixir tailored for
  numerical computations. For example, the following function:

      defn add_and_mult(a, b, c) do
        a * b + c
      end

  will work with scalars, vector, matrices, and n-dimensional
  tensors, Depending on your backend of choice, the code can even
  be JIT-compiled or AOT-compiled and run either on the CPU or GPU.

  `defn` is a subset of Elixir since it replaces Elixir's `Kernel`
  by `Nx.Defn.Kernel`. `Nx.Defn.Kernel` provides tensor-aware
  operators, such as `+`, `-`, etc, while also preserving many
  high-level constructs known to Elixir developers, such as pipe
  operator, aliases, conditionals, pattern-matching, and more.
  Please consult its documentation for a complete reference.

  You can also call functions from the `Nx` module directly inside
  `defn`. For example, the code above can also be written as:

      defn add_and_mult(a, b, c) do
        a
        |> Nx.multipy(b)
        |> Nx.add(c)
      end

  A handful of `Nx` functions are not supported in `defn` but
  generally speaking you don't have to worry about them. If you
  use any unsupported API, you will get an error at compilation
  time.

  Calling other functions is possible, as long as they are
  implemented with `defn`.

  ## Inputs and outputs types

  `defn` functions can receive either tuples, numbers, or tensors
  as inputs.

  Tensors are passed in as is. If a number is given, it is automatically
  cast to tensor. If a tuple is expected, it must be matched on the
  function head itself. For example, do this:

      defn add_tuple({a, b}), do: a + b

  Do not do this:

      defn add_tuple(tuple) do
        {a, b} = tuple
        a + b
      end

  `defn` functions can return tensors or tuples of tensors. Note tensors
  are returned even for numerical constants. For example, this function:

      defn just_two(), do: 2

  will return the same as `Nx.tensor(2)`.

  ## Compilers

  The power of `Nx.Defn` is given by its compilers. The default
  compiler is the `Nx.Defn` module itself, which executes the code
  in pure Elixir. However, you can use module attributes to specify
  how a `defn` function will behave. For example, assuming you
  are using the `EXLA` compiler:

      @defn_compiler {EXLA, client: :host}
      defn add_and_mult(a, b, c) do
        a * b + c
      end

  To set the compiler for the all definitions, you can set the
  `@default_defn_compiler` attribute:

      @default_defn_compiler {EXLA, client: :cuda}

  """

  ## Default compiler backend

  @behaviour Nx.Defn.Compiler

  @impl true
  def __compile__(_env, _kind, vars, fun, []) do
    fun
    |> apply(vars)
    |> to_result(vars, %{})
    |> elem(0)
  end

  defp eval(%Nx.Tensor{data: %Nx.Defn.Expr{op: :fun, args: [_, _, _, fun]}}, _vars, cache) do
    {fun, cache}
  end

  defp eval(%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [i]}}, vars, cache) do
    {Enum.fetch!(vars, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Nx.Defn.Expr{op: :tensor, args: [t]}}, _vars, cache) do
    {t, cache}
  end

  defp eval(%Nx.Tensor{data: %Nx.Defn.Expr{id: id, op: op, args: args}} = ans, vars, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {args, cache} = Enum.map_reduce(args, cache, &eval(&1, vars, &2))
        res = apply(Nx.Shared.find_impl!(args), op, [ans | args])
        {res, Map.put(cache, id, res)}
    end
  end

  defp eval(other, _vars, cache) do
    {other, cache}
  end

  defp to_result(tuple, vars, cache) when is_tuple(tuple) do
    {args, cache} =
      tuple
      |> Tuple.to_list()
      |> Enum.map_reduce(cache, &to_result(&1, vars, &2))

    {List.to_tuple(args), cache}
  end

  defp to_result(other, vars, cache) do
    {expr, cache} = eval(other, vars, cache)
    {Nx.tensor(expr), cache}
  end

  ## Public API

  @doc """
  Defines a public numerical function.
  """
  defmacro defn(call, do: block) do
    define(:def, call, block, __CALLER__)
  end

  @doc """
  Defines a private numerical function.

  Private numerical functions are always inlined by
  their callers at compilation time. This happens to
  all local function calls within `defn`.
  """
  defmacro defnp(call, do: block) do
    define(:defp, call, block, __CALLER__)
  end

  ## Callbacks

  defp define(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    # Note name here is not necessarily an atom due to unquote(name) support
    {name, args} = decompose_call!(kind, call, env)
    assert_no_defaults!(kind, args, env)
    arity = length(args)

    quote do
      unquote(__MODULE__).__define__(__MODULE__, unquote(kind), unquote(name), unquote(arity))

      unquote(kind)(unquote(call)) do
        use Nx.Defn.Kernel
        unquote(block)
      end
    end
  end

  defp decompose_call!(_kind, {{:unquote, _, [name]}, _, args}, _env) do
    {name, args}
  end

  defp decompose_call!(kind, call, env) do
    case Macro.decompose_call(call) do
      {name, args} ->
        {name, args}

      :error ->
        compile_error!(
          env,
          "first argument of #{kind}n must be a call, got: #{Macro.to_string(call)}"
        )
    end
  end

  defp assert_no_guards!(kind, {:when, _, _}, env) do
    compile_error!(env, "guards are not supported by #{kind}n")
  end

  defp assert_no_guards!(_kind, _call, _env), do: :ok

  defp assert_no_defaults!(kind, call, env) do
    if default = Enum.find(call, &match?({:\\, _, _}, &1)) do
      compile_error!(
        env,
        "default arguments are not supported by #{kind}n, got: #{Macro.to_string(default)}"
      )
    end
  end

  # Internal attributes
  @exports_key :__next_exports__

  # Per-defn attributes
  @defn_compiler :defn_compiler

  # Module attributes
  @default_defn_compiler :default_defn_compiler

  @doc false
  def __define__(module, kind, name, arity) do
    exports =
      if exports = Module.get_attribute(module, @exports_key) do
        exports
      else
        Module.put_attribute(module, :before_compile, __MODULE__)
        %{}
      end

    compiler =
      Module.delete_attribute(module, @defn_compiler) ||
        Module.get_attribute(module, @default_defn_compiler) ||
        __MODULE__

    compiler = normalize_compiler!(compiler)

    exports = Map.put(exports, {name, arity}, %{kind: kind, compiler: compiler})
    Module.put_attribute(module, @exports_key, exports)
    :ok
  end

  defp normalize_compiler!(atom) when is_atom(atom), do: {atom, []}
  defp normalize_compiler!({atom, term}) when is_atom(atom), do: {atom, term}

  defp normalize_compiler!(other) do
    raise ArgumentError,
          "expected @defn_compiler/@default_defn_compiler to be an atom or " <>
            "a tuple with an atom as first element, got: #{inspect(other)}"
  end

  defp compile_error!(env, description) do
    raise CompileError, line: env.line, file: env.file, description: description
  end

  @doc false
  defmacro __before_compile__(env) do
    exports = Module.get_attribute(env.module, @exports_key)
    Nx.Defn.Compiler.__compile__(env, exports)
  end
end
