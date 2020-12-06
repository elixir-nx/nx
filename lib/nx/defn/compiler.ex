defmodule Nx.Defn.Compiler do
  @moduledoc """
  The specification and helper functions for custom `defn` compilers.
  """

  @forbidden_nx_functions [tensor: 1, tensor: 2, device_read: 1, device_deallocate: 1] ++
                            [device_transfer: 1, device_transfer: 2, device_transfer: 3]

  @known_keywords [:type, :axis]

  @doc """
  The callback required to be implemented for each compiler.

  It receives the function `kind`, the function `metadata`,
  the arguments (where each argument is the variable AST),
  the AST, and the compiler options. The AST is guaranteed
  to be valid Elixir code. It must return valid Elixir AST.

  Each variable in the AST has a :counter metadata which is
  guaranteed to uniquely identify each variable.

  The callback uses double underscores so it can be defined
  at root modules without affecting the module's main API.

  ## Nodes

    * `{var_atom_name, meta, context_atom}` - variables

    * `{:__block__, meta, [expr]}` - blocks

    * `{left, right}` and `{:{}, _, [elem]}` - tuples

    * `{:=, meta, [left, right]}` - pattern matching where the
      left side has been normalized to be either a variable or
      a `{:{}, _, [var]}`

    * `{:%{}, meta, [{key, value}]}` - a literal `Nx.Tensor`

    * `{{:., dot_meta, [Nx, fun]}, meta, [arg]}` - call to the `Nx`
      module (most functions are allowed unless the ones listed in
      the "Unsupported Nx functions" section)

  ## Unsupported Nx functions

  For completeness, here is a list of all `Nx` functions that
  are not allowed in `defn`:

  #{for {f, a} <- @forbidden_nx_functions, do: "  * `#{Exception.format_mfa(Nx, f, a)}`\n"}
  """
  @callback __compile__(
              kind :: :def | :defp,
              metadata :: keyword,
              name :: atom,
              args :: [Macro.t()],
              ast :: Macro.t(),
              opts :: keyword
            ) :: Macro.t()

  # The compiler has four passes.
  #
  # 1. Normalize and validate the AST written by the user.
  #
  # 2. Expand patterns and inline all local functions. This is
  #    the result stored in each module for cross-module inlining.
  #
  # 3. Inline remote calls.
  #
  # 4. Invoke the compiler chosen by the user.
  #
  # The user compiler may have multiple passes and it has the ability
  # to run said additional passes either at runtime or compilation time.
  @doc false
  def compile(%Macro.Env{module: module, file: file, line: line}, exports) do
    {:module, Nx} = Code.ensure_loaded(Nx)
    cache = for {def, _meta} <- exports, into: %{}, do: {def, false}
    public = for {def, %{kind: :def} = meta} <- exports, do: {def, meta}

    state = %{
      module: module,
      file: file,
      stack: [],
      cache: cache,
      function: nil,
      line: line,
      version: 0
    }

    {quoted, state} = Enum.map_reduce(public, state, &compile_each/2)

    to_delete =
      for {def, stored} when stored != false <- state.cache do
        quote do
          Nx.Defn.Module.delete_definition(__MODULE__, unquote(def))
        end
      end

    {:__block__, [], to_delete ++ quoted}
  end

  defp compile_each({{name, arity} = def, def_meta}, state) do
    {{kind, meta, args, ast}, state} = get_cached_definition(def, state)
    {def_module, def_opts} = def_meta.compiler
    defn_ast = Macro.escape({meta, args, ast})
    compiled_ast = def_module.__compile__(kind, meta, name, args, ast, def_opts)

    quoted =
      quote do
        def unquote(name)(unquote_splicing(args)), do: unquote(compiled_ast)
        def __defn__(unquote(name), unquote(arity)), do: unquote(defn_ast)
      end

    {quoted, state}
  end

  defp get_cached_definition(def, state) do
    case state.cache do
      %{^def => false} -> get_and_cache_definition(def, state)
      %{^def => stored} -> {stored, state}
      %{} -> :none
    end
  end

  defp get_and_cache_definition(def, state) do
    {:v1, kind, meta, clauses} = Nx.Defn.Module.get_definition(state.module, def)

    with_call(meta, def, def, state, fn state ->
      case clauses do
        [] ->
          compile_error!(meta, state, "cannot have #{kind}n without clauses")

        [{meta, args, [], ast}] ->
          {args, state} = normalize_list(args, state)
          {ast, state} = normalize_block(ast, meta, state)
          {ast, state} = expand(ast, state)
          result = {kind, [max_counter: state.version] ++ meta, args, ast}
          state = put_in(state.cache[def], result)
          {result, state}

        [_, _ | _] ->
          compile_error!(meta, state, "cannot compile #{kind}n with multiple clauses")
      end
    end)
  end

  defp with_call(meta, def, call, state, fun) do
    %{function: previous_def, stack: previous_stack, line: previous_line} = state
    line = meta[:line] || previous_line
    {result, state} = fun.(%{state | function: def, stack: [call | previous_stack], line: line})
    {result, %{state | function: previous_def, stack: previous_stack, line: previous_line}}
  end

  ## Normalization

  defp normalize({:__block__, meta, _} = block, state) do
    normalize_block(block, meta, state)
  end

  defp normalize({:{}, meta, args}, state) do
    {args, state} = normalize_list(args, state)
    {{:{}, meta, args}, state}
  end

  defp normalize({:=, meta, [_, _] = args}, state) do
    {args, state} = normalize_list(args, state)
    {{:=, meta, args}, state}
  end

  defp normalize(
         {:%{}, meta, [__struct__: Nx.Tensor, data: {device, data}, shape: shape, type: {_, _}]} =
           tensor,
         state
       )
       when is_tuple(shape) do
    if device == Nx.BitStringDevice and is_bitstring(data) do
      {tensor, state}
    else
      compile_error!(
        meta,
        state,
        "defn expects a tensor allocated on Nx.BitStringDevice as a constant/module attribute, got: " <>
          inspect(device)
      )
    end
  end

  defp normalize({name, meta, args}, state)
       when is_atom(name) and is_list(args) and
              name not in [:%, :%{}, :^, :<<>>] do
    {args, state} = normalize_list(args, state)
    {{:__local__, meta, [name | args]}, state}
  end

  defp normalize({name, meta, ctx}, state) when is_atom(name) and is_atom(ctx) do
    {version, meta} = Keyword.pop!(meta, :version)
    state = update_in(state.version, &max(&1, version))
    {{name, [counter: version, generated: true] ++ meta, ctx}, state}
  end

  defp normalize({{:., _, [Nx, name]} = call, meta, args}, state) do
    arity = length(args)

    unless function_exported?(Nx, name, arity) do
      compile_error!(meta, state, "undefined function Nx.#{name}/#{arity}")
    end

    if {name, arity} in @forbidden_nx_functions do
      compile_error!(meta, state, "Nx.#{name}/#{arity} is not allowed inside defn")
    end

    {args, state} = normalize_list(args, state)
    {{call, meta, args}, state}
  end

  defp normalize({left, right}, state) do
    {left, state} = normalize(left, state)
    {right, state} = normalize(right, state)
    {{left, right}, state}
  end

  defp normalize(list, state) when is_list(list) do
    cond do
      not Keyword.keyword?(list) ->
        compile_error!(
          [],
          state,
          "invalid numerical expression: #{Macro.to_string(list)} (only keyword lists are allowed)"
        )

      not Enum.all?(list, fn {k, _} -> k in @known_keywords end) ->
        compile_error!(
          [],
          state,
          "invalid numerical expression: #{Macro.to_string(list)} (the only allowed keys " <>
            "in keyword lists are: #{Enum.map_join(@known_keywords, ", ", &inspect/1)})"
        )

      true ->
        normalize_list(list, state)
    end
  end

  defp normalize(literal, state) when is_number(literal) or is_atom(literal) do
    {literal, state}
  end

  defp normalize(expr, state) do
    compile_error!(
      maybe_meta(expr),
      state,
      "invalid numerical expression: #{Macro.to_string(expr)}"
    )
  end

  defp normalize_list(list, state) do
    Enum.map_reduce(list, state, &normalize/2)
  end

  ## Normalize block handling

  defp normalize_block({:__block__, meta, exprs}, _meta, state) do
    {exprs, state} = normalize_block(exprs, [], meta, state)
    {{:__block__, meta, exprs}, state}
  end

  defp normalize_block(expr, meta, state) do
    {[expr], state} = normalize_block([expr], [], meta, state)
    {expr, state}
  end

  defp normalize_block([{:__block__, _, head} | rest], acc, meta, state) do
    normalize_block(head ++ rest, acc, meta, state)
  end

  defp normalize_block([nil], acc, meta, state) do
    compile_warn(meta, state, "body has nil return type, 0 will be returned instead")
    {Enum.reverse([0 | acc]), state}
  end

  defp normalize_block([last], acc, _meta, state) do
    {last, state} = normalize(last, state)
    {Enum.reverse([last | acc]), state}
  end

  # alias, require, import are expanded to atoms, so we ignore those.
  defp normalize_block([atom | rest], acc, meta, state) when is_atom(atom) do
    normalize_block(rest, acc, meta, state)
  end

  defp normalize_block([head | rest], acc, meta, state) do
    {head, state} = normalize(head, state)
    normalize_block(rest, [head | acc], meta, state)
  end

  ## Expansion and Local inlining

  defp expand({:=, meta, [left, right]}, state) do
    {patterns, vars, right} = expand_assign(meta, left, right, state)
    {right, state} = expand(right, state)

    {[var | vars], state} =
      case vars do
        [] ->
          {var, state} = new_var(state)
          {[var], state}

        _ ->
          {vars, state}
      end

    {patterns, nested, state} = expand_nested_patterns(patterns, state)

    exprs =
      [{:=, meta, [var, right]}] ++
        Enum.map(vars, &{:=, meta, [&1, var]}) ++
        Enum.map(patterns, &{:=, meta, [&1, var]}) ++
        nested

    case exprs do
      [expr] -> {expr, state}
      _ -> {{:__block__, meta, exprs}, state}
    end
  end

  defp expand({:__local__, meta, [name | args]}, state) do
    arity = length(args)

    if {name, arity} in state.stack do
      {caller_name, caller_arity} = state.function

      compile_error!(
        meta,
        state,
        "#{name}/#{arity} is being called recursively by #{caller_name}/#{caller_arity}, " <>
          "defn does not allow recursive definitions"
      )
    end

    case get_cached_definition({name, arity}, state) do
      {{_kind, _meta, vars, ast}, state} ->
        version = state.version + 1

        {{vars, ast}, max_version} =
          Macro.prewalk({vars, ast}, version, fn
            {var, meta, ctx}, max_version when is_atom(var) and is_atom(ctx) ->
              {var_version, meta} =
                Keyword.get_and_update!(meta, :counter, &{&1 + version, &1 + version})

              {{var, meta, ctx}, max(max_version, var_version)}

            node, max_version ->
              {node, max_version}
          end)

        assigns =
          vars
          |> Enum.zip(args)
          |> Enum.map(fn {var, arg} -> {:=, meta, [var, arg]} end)

        {{:__block__, meta, assigns ++ [ast]}, %{state | version: max_version}}

      :none ->
        compile_error!(
          meta,
          state,
          "cannot invoke #{name}/#{arity} because it was not defined with defn"
        )
    end
  end

  defp expand({name, meta, args}, state) do
    {args, state} = expand(args, state)
    {{name, meta, args}, state}
  end

  defp expand({left, right}, state) do
    {left, state} = expand(left, state)
    {right, state} = expand(right, state)
    {{left, right}, state}
  end

  defp expand(list, state) when is_list(list) do
    Enum.map_reduce(list, state, &expand/2)
  end

  defp expand(other, state) do
    {other, state}
  end

  ### Pattern matching

  defp expand_assign(outer_meta, outer_left, {:=, inner_meta, [inner_left, expr]}, state) do
    {patterns, vars, expr} = expand_assign(inner_meta, inner_left, expr, state)
    expand_pattern(outer_meta, outer_left, patterns, vars, expr, state)
  end

  defp expand_assign(meta, left, expr, state) do
    expand_pattern(meta, left, [], [], expr, state)
  end

  defp expand_pattern(_, {:=, meta, [left, right]}, patterns, vars, expr, state) do
    {patterns, vars, expr} = expand_pattern(meta, right, patterns, vars, expr, state)
    expand_pattern(meta, left, patterns, vars, expr, state)
  end

  defp expand_pattern(_meta, {name, _, ctx} = var, patterns, vars, expr, _state)
       when is_atom(name) and is_atom(ctx) do
    {patterns, [var | vars], expr}
  end

  defp expand_pattern(meta, pattern, patterns, vars, expr, state) do
    pattern = validate_pattern!(meta, pattern, state)
    {[pattern | patterns], vars, expr}
  end

  defp validate_pattern!(meta, {left, right}, _state), do: {:{}, meta, [left, right]}
  defp validate_pattern!(_meta, {:{}, _, _} = tuple,  _state), do: tuple

  defp validate_pattern!(_meta, {name, _, ctx} = var, _state)
       when is_atom(name) and is_atom(ctx),
       do: var

  defp validate_pattern!(meta, expr, state) do
    compile_error!(
      meta,
      state,
      "defn can only pattern match on variables or tuples, got: #{Macro.to_string(expr)}"
    )
  end

  defp expand_nested_patterns(patterns, state) do
    {patterns, {nested, state}} =
      Enum.map_reduce(patterns, {[], state}, fn {:{}, meta, args}, {nested, state} ->
        {args, nested, state} = expand_nested_patterns(args, meta, [], nested, state)
        {{:{}, meta, args}, {nested, state}}
      end)

    {nested, state} = expand(nested, state)
    {patterns, nested, state}
  end

  defp expand_nested_patterns([{name, _, ctx} = var | args], meta, acc, nested, state)
       when is_atom(name) and is_atom(ctx) do
    expand_nested_patterns(args, meta, [var | acc], nested, state)
  end

  defp expand_nested_patterns([expr | args], meta, acc, nested, state) do
    {var, state} = new_var(state)
    expand_nested_patterns(args, meta, [var | acc], [{:=, meta, [expr, var]} | nested], state)
  end

  defp expand_nested_patterns([], _meta, acc, nested, state) do
    {Enum.reverse(acc), nested, state}
  end

  ## Shared helpers

  defp new_var(state) do
    counter = state.version + 1
    {{:nvar, [counter: counter], __MODULE__}, %{state | version: counter}}
  end

  defp maybe_meta({_, meta, _}), do: meta
  defp maybe_meta(_), do: []

  defp compile_warn(meta, state, message) do
    {name, arity} = state.function
    line = meta[:line] || state.line
    file = String.to_charlist(state.file)
    stacktrace = [{state.module, name, arity, line: line, file: file}]
    IO.warn(message, stacktrace)
  end

  defp compile_error!(meta, state, description) do
    line = meta[:line] || state.line
    raise CompileError, line: line, file: state.file, description: description
  end
end
