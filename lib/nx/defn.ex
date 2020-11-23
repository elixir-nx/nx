defmodule Nx.Defn do
  @moduledoc """
  Compile-time numerical definitions.

  A numerical function is a subset of Elixir tailored for
  numerical computations. For example, the following function:

      defn add_and_mult(a, b, c) do
        a * b + c
      end

  Can be called with scalars, vectors, matrices, or n-dimensional
  tensors. Depending on your backend of choice, the code can even
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

  The only `Nx` functions not supported in `defn` are conversion
  functions (such as `Nx.to_bitstring/1`) and device transfer
  functions as they are meant for interfacing between Elixir and
  the numerical world.

  Calling other functions is possible, as long as they are implemented
  with `defn`. At the moment, only calling local functions is supported
  by remote functions are coming next.

  ## Compilers

  The power of `Nx.Defn` is given by its compilers. The default
  compiler is the `Nx.Defn` module itself, which executes the code
  in pure Elixir. However, you can use module attributes to specify
  how a `defn` function will behave. For example, assuming you
  are using the `Exla` compiler:

      @defn_compiler {Exla, device: :cpu}
      defn add_and_mult(a, b, c) do
        a * b + c
      end

  To set the compiler for the all definitions, you can set the
  `@default_defn_compiler` attribute:

      @default_defn_compiler {Exla, device: :gpu}

  """

  ## Default compiler backend

  @behaviour Nx.Defn.Compiler

  @impl true
  def __compile__(_kind, _meta, args, ast, []) do
    quote do
      unquote(__MODULE__).__validate__!(unquote(args))
      unquote(ast)
    end
  end

  @doc false
  def __validate__!(args) do
    for arg <- args,
        not is_number(arg),
        not match?(%Nx.Tensor{}, arg) do
      raise ArgumentError, "defn functions expects either numbers or %Nx.Tensor{} as arguments"
    end

    :ok
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

  # TODO: Support default arguments
  # TODO: Support cross module calls
  defp define(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    {name, args} = decompose_call!(kind, call, env)
    assert_no_defaults!(kind, args, env)
    assert_only_vars!(kind, args, env)
    arity = length(args)

    quote do
      unquote(__MODULE__).__define__(__MODULE__, unquote(kind), unquote(name), unquote(arity))

      unquote(kind)(unquote(name)(unquote_splicing(args))) do
        use Nx.Defn.Kernel
        unquote(block)
      end
    end
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

  defp assert_only_vars!(kind, args, state) do
    if expr = Enum.find(args, &(not match?({var, _, ctx} when is_atom(var) and is_atom(ctx), &1))) do
      compile_error!(
        state,
        "only variables are allowed as arguments in #{kind}n, got: #{Macro.to_string(expr)}"
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
      Module.get_attribute(module, @defn_compiler) ||
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
    Nx.Defn.Compiler.compile(env, exports)
  end
end
