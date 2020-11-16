defmodule NEXT.Kernel do
  @moduledoc """
  The API available inside `defn` blocks.
  """

  def a + b, do: :erlang.+(a, b)
  def add(a, b), do: :erlang.+(a, b)

  def a - b, do: :erlang.-(a, b)
  def subtract(a, b), do: :erlang.-(a, b)
end

defmodule NEXT.Module do
  @moduledoc false

  # TODO: send those as pull requests to Elixir.

  @doc """
  Returns the definition for the name-arity pair.
  
  It returns a tuple with the `version`, the `kind`,
  the definition `metadata`, and a list with each clause.
  Each clause is a four-element tuple with metadata,
  the arguments, the guards, and the clause AST.
  
  The clauses are returned in the expanded AST format,
  which is a subset of Elixir's AST but already normalized.
  This makes it a useful AST for analyzing code but it
  cannot be reinjected into the module as it may have
  lost some of its original context. Given this AST
  representation is mostly internal, it is versioned
  and it may change at any time. Therefore, **use this
  API with caution**.
  """
  # @spec get_definition(module, definition) ::
  #         {:v1, kind, meta :: keyword,
  #          [{meta :: keyword, arguments :: [Macro.t()], guards :: [Macro.t()], Macro.t()}]}
  def get_definition(module, {name, arity})
      when is_atom(module) and is_atom(name) and is_integer(arity) do
    # assert_not_compiled!(__ENV__.function, module, @extra_error_msg_definitions_in)
    {set, bag} = data_tables_for(module)

    case :ets.lookup(set, {:def, {name, arity}}) do
      [{_key, kind, meta, _, _, _}] ->
        {:v1, kind, meta, bag_lookup_element(bag, {:clauses, {name, arity}}, 2)}

      [] ->
        nil
    end
  end

  @doc """
  Deletes a definition from a module.

  It returns true if the definition exists and it was removed,
  otherwise it returns false.
  """
  # @spec delete_definition(module, definition) :: boolean()
  def delete_definition(module, {name, arity})
      when is_atom(module) and is_atom(name) and is_integer(arity) do
    # assert_not_readonly!(__ENV__.function, module)

    case :elixir_def.take_definition(module, {name, arity}) do
      false ->
        false
            
      _ ->
        :elixir_locals.yank({name, arity}, module)
        true
    end
  end

  ## These helpers already exist in Elixir's lib/module.ex

  defp data_tables_for(module) do
    :elixir_module.data_tables(module)
  end

  defp bag_lookup_element(table, key, pos) do
    :ets.lookup_element(table, key, pos)
  catch
    :error, :badarg -> []
  end
end

defmodule NEXT do
  @moduledoc """
  Numerical EliXir Transforms.
  """

  @defs_key :__next_defs__

  # TODO: Support default arguments
  # TODO: Support multiple clauses
  # TODO: Support private
  # TODO: Support cross module calls
  # TODO: Support pipe
  # TODO: Support guards
  # TODO: Support if
  @doc """
  Defines a numerical function.

  A numerical function is a subset of Elixir tailored for
  numerical computations. For example, the following functions:

      defn add_and_mult(a, b, c) do
        a * b + c
      end

  Can be called with scalars, vectors, matrices or n-dimensional
  tensors. Depending on your backend of choice, the code can even
  be JIT-compiled or AOT compiled.

  This works by replacing Elixir's `Kernel` by `NEXT.Kernel`. In
  other words, you can see all functionality available inside `defn`
  by consulting the docs to `NEXT.Kernel`.
  """
  defmacro defn(call, do: block) do
    define(:def, call, block, __CALLER__)
  end

  defp define(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    {name, args} = decompose_call!(kind, call, env)
    assert_no_defaults!(kind, args, env)
    arity = length(args)

    quote do
      unquote(__MODULE__).__define__(__MODULE__, unquote(kind), unquote(name), unquote(arity))

      Kernel.unquote(kind)(unquote(name)(unquote_splicing(args))) do
        import Kernel, only: []
        import NEXT.Kernel
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
          [],
          env,
          "first argument of #{kind}n must be a call, got: #{Macro.to_string(call)}"
        )
    end
  end

  defp assert_no_guards!(kind, {:when, _, _}, env) do
    compile_error!([], env, "guards are not supported by #{kind}n")
  end

  defp assert_no_guards!(_kind, _call, _env), do: :ok

  defp assert_no_defaults!(kind, call, env) do
    if Enum.any?(call, &match?({:\\, _, _}, &1)) do
      compile_error!(
        [],
        env,
        "default arguments are not supported by #{kind}n, got: #{Macro.to_string(call)}"
      )
    end
  end

  @doc false
  def __define__(module, kind, name, arity) do
    defs =
      if defs = Module.get_attribute(module, @defs_key, nil) do
        defs
      else
        Module.put_attribute(module, :before_compile, __MODULE__)
        []
      end

    defs = if kind == :def, do: [{name, arity} | defs], else: defs
    Module.put_attribute(module, @defs_key, defs)
    :ok
  end

  ## Compilation

  @doc false
  defmacro __before_compile__(%Macro.Env{module: module, file: file}) do
    defs = Module.get_attribute(module, @defs_key)
    state = %{module: module, file: file}
    {defs, _state} = Enum.map_reduce(defs, state, &compile/2)
    {:__block__, [], defs}
  end

  defp compile({name, _arity} = def, state) do
    {:v1, kind, meta, clauses} = NEXT.Module.get_definition(state.module, def)

    case clauses do
      [] ->
        {:ok, state}

      [{meta, args, [], ast}] ->
        assert_only_vars!(kind, meta, args, state)
        binding = Enum.map(args, &{elem(&1, 0), &1})

        quoted =
          quote do
            NEXT.Module.delete_definition(__MODULE__, unquote(def))

            def unquote(name)(unquote_splicing(args)) do
              {unquote(Macro.escape(ast)), unquote(binding)}
            end
          end

        {quoted, state}

      [_, _ | _] ->
        compile_error!(meta, state, "cannot compile #{kind}n with multiple clauses")
    end
  end

  defp assert_only_vars!(kind, meta, args, state) do
    unless Enum.all?(args, &match?({var, _, ctx} when is_atom(var) and is_atom(ctx), &1)) do
      compile_error!(
        meta,
        state,
        "only variables are allowed as arguments in #{kind}n, got: #{Macro.to_string(args)}"
      )
    end
  end

  ## Shared helpers

  defp compile_error!(meta, env_or_state, message) do
    line = meta[:line] || env_or_state.line
    raise CompileError, line: line, file: env_or_state.file, message: message
  end
end
