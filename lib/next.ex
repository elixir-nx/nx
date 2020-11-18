defmodule Nx do
  @moduledoc """
  Numerical Elixir.

  A collection of functions and data types to work
  with Numerical Elixir.
  """

  def add(a, b) when is_integer(a) and is_integer(b), do: :erlang.+(a, b)
  def add(a, b) when is_list(a) and is_integer(b), do: Enum.map(a, &:erlang.+(&1, b))
end

defmodule NEXT.Kernel do
  @moduledoc """
  The API available inside `defn` blocks.
  """

  defmacro a + b do
    quote do: Nx.add(unquote(a), unquote(b))
  end
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

  @exports_key :__next_exports__

  # TODO: Support default arguments
  # TODO: Support multiple clauses
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

  @doc """
  Defines a private numerical function.

  Like any numerical function, it is always inlined
  by its callers and then removed and made innaccessible
  at compilation time.
  """
  defmacro defnp(call, do: block) do
    define(:defp, call, block, __CALLER__)
  end

  defp define(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    {name, args} = decompose_call!(kind, call, env)
    assert_no_defaults!(kind, args, env)
    arity = length(args)

    quote do
      unquote(__MODULE__).__define__(__MODULE__, unquote(kind), unquote(name), unquote(arity))

      unquote(kind)(unquote(name)(unquote_splicing(args))) do
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

  ## Compilation

  @doc false
  defmacro __before_compile__(%Macro.Env{module: module, file: file, line: line}) do
    exports = Module.get_attribute(module, @exports_key)
    defs = for {def, _value} <- exports, into: %{}, do: {def, false}
    state = %{module: module, file: file, stack: [], defs: defs, function: nil, line: line}

    public = for {def, :def} <- exports, do: def
    {quoted, state} = Enum.map_reduce(public, state, &compile/2)

    to_delete =
      for {def, stored} when stored != false <- state.defs do
        quote do
          NEXT.Module.delete_definition(__MODULE__, unquote(def))
        end
      end

    {:__block__, [], to_delete ++ quoted}
  end

  defp compile({name, _arity} = def, state) do
    state = %{state | function: def}
    {{_kind, _meta, args, ast}, state} = get_cached_definition(def, [], state)

    binding =
      Enum.map(args, fn {var, meta, ctx} -> {var, {var, [generated: true] ++ meta, ctx}} end)

    quoted =
      quote do
        def unquote(name)(unquote_splicing(args)) do
          {unquote(Macro.escape(ast)), unquote(binding)}
        end
      end

    {quoted, state}
  end

  defp get_cached_definition({name, arity} = def, meta, state) do
    case state.defs do
      %{^def => false} ->
        get_and_cache_definition(def, state)

      %{^def => stored} ->
        {stored, state}

      %{} ->
        compile_error!(
          meta,
          state,
          "cannot invoke #{name}/#{arity} because it was not defined with defn"
        )
    end
  end

  defp get_and_cache_definition(def, state) do
    with_function(def, state, fn state ->
      {:v1, kind, meta, clauses} = NEXT.Module.get_definition(state.module, def)

      case clauses do
        [] ->
          compile_error!(meta, state, "cannot have #{kind}n without clauses")

        [{meta, args, [], ast}] ->
          assert_only_vars!(kind, meta, args, state)
          {ast, %{version: version}} = normalize_block(ast, meta, %{version: 0}, state)
          result = {kind, [max_version: version] ++ meta, args, ast}
          state = put_in(state.defs[def], result)
          {result, state}

        [_, _ | _] ->
          compile_error!(meta, state, "cannot compile #{kind}n with multiple clauses")
      end
    end)
  end

  defp with_function(def, %{function: previous_def} = state, fun) do
    {result, state} = fun.(%{state | function: def})
    {result, %{state | function: previous_def}}
  end

  defp normalize({name, meta, ctx} = var, context, _state) when is_atom(name) and is_atom(ctx) do
    version = Keyword.fetch!(meta, :version)
    context = update_in(context.version, &max(&1, version))
    {var, context}
  end

  defp normalize({{:., _, [Nx, _]} = call, meta, args}, context, state) do
    {args, context} = normalize_list(args, context, state)
    {{call, meta, args}, context}
  end

  defp normalize({:__block__, meta, _} = block, context, state) do
    normalize_block(block, meta, context, state)
  end

  defp normalize(expr, _context, state) do
    compile_error!(
      maybe_meta(expr),
      state,
      "invalid numerical expression: #{Macro.to_string(expr)}"
    )
  end

  defp normalize_list(list, context, state) do
    Enum.map_reduce(list, context, &normalize(&1, &2, state))
  end

  ## Block handling

  defp normalize_block({:__block__, meta, exprs}, _meta, context, state) do
    {exprs, context} = normalize_block(exprs, [], meta, context, state)
    {{:__block__, meta, exprs}, context}
  end

  defp normalize_block(expr, meta, context, state) do
    {[expr], context} = normalize_block([expr], [], meta, context, state)
    {expr, context}
  end

  defp normalize_block([nil], acc, meta, context, state) do
    compile_warn(meta, state, "body has nil return type, -1 will be returned instead")
    {Enum.reverse([-1 | acc]), context}
  end

  defp normalize_block([last], acc, _meta, context, state) do
    {last, context} = normalize(last, context, state)
    {Enum.reverse([last | acc]), context}
  end

  # alias, require, import are expanded to atoms, so we ignore those.
  defp normalize_block([atom | rest], acc, meta, context, state) when is_atom(atom) do
    normalize_block(rest, acc, meta, context, state)
  end

  defp normalize_block([head | rest], acc, meta, context, state) do
    {head, context} = normalize(head, context, state)
    normalize_block(rest, [head | acc], meta, context, state)
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

  defp maybe_meta({_, meta, _}), do: meta
  defp maybe_meta(_), do: []

  defp compile_warn(meta, env_or_state, message) do
    {name, arity} = env_or_state.function
    line = meta[:line] || env_or_state.line
    file = String.to_charlist(env_or_state.file)
    stacktrace = [{env_or_state.module, name, arity, line: line, file: file}]
    IO.warn(message, stacktrace)
  end

  defp compile_error!(meta, env_or_state, description) do
    line = meta[:line] || env_or_state.line
    raise CompileError, line: line, file: env_or_state.file, description: description
  end
end
