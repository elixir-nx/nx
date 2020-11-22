defmodule Nx.Defn do
  @moduledoc """
  Compile-time numerical definitions.

  A numerical function is a subset of Elixir tailored for
  numerical computations. For example, the following function:

      defn add_and_mult(a, b, c) do
        a * b + c
      end

  Can be called with scalars, vectors, matrices or n-dimensional
  tensors. Depending on your backend of choice, the code can even
  be JIT-compiled or AOT-compiled and run either on the CPU or GPU.

  `defn` is a subset of Elixir since it replaces Elixir's `Kernel`
  by `Nx.Defn.Kernel`. You can also call most functions from the
  `Nx` module directly inside `defn`. For example, the above can
  also be written as:

      defn add_and_mult(a, b, c) do
        a
        |> Nx.multipy(b)
        |> Nx.add(c)
      end

  The only `Nx` functions not supported in `defn` are conversion
  functions (such as `Nx.to_bitstring/1`) and device transfer
  functions as they are meant for interfacing between Elixir and
  the numerical world.
  """

  @exports_key :__next_exports__

  # TODO: Support default arguments
  # TODO: Support cross module calls
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

  defp define(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    {name, args} = decompose_call!(kind, call, env)
    assert_no_defaults!(kind, args, env)
    assert_only_vars!(kind, args, env)
    arity = length(args)

    quote do
      unquote(__MODULE__).__define__(__MODULE__, unquote(kind), unquote(name), unquote(arity))

      unquote(kind)(unquote(name)(unquote_splicing(args))) do
        import Kernel, only: []
        import Nx.Defn.Kernel
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

  @doc false
  def __define__(module, kind, name, arity) do
    exports =
      if exports = Module.get_attribute(module, @exports_key, nil) do
        exports
      else
        Module.put_attribute(module, :before_compile, __MODULE__)
        %{}
      end

    exports = Map.put(exports, {name, arity}, kind)
    Module.put_attribute(module, @exports_key, exports)
    :ok
  end

  @doc false
  defmacro __before_compile__(env) do
    exports = Module.get_attribute(env.module, @exports_key)
    Nx.Defn.Compiler.compile(env, exports)
  end

  ## Helpers

  defp compile_error!(env, description) do
    raise CompileError, line: env.line, file: env.file, description: description
  end
end
