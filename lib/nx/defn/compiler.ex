defmodule Nx.Defn.Compiler do
  @moduledoc false

  # The compiler has four passes.
  #
  # 1. Normalize and validate the AST written by the user.
  #
  # 2. Inline all local functions. This is stored in each module
  #    for cross-module inlining.
  #
  # 3. Inline of remote functions.
  #
  # 4. Invoke the compiler chosen by the user
  #
  # The user compiler may have multiple passes and it has the ability
  # to run said additional passes either at runtime or compilation time.
  def compile(%Macro.Env{module: module, file: file, line: line}, exports) do
    defs = for {def, _value} <- exports, into: %{}, do: {def, false}

    state = %{
      module: module,
      file: file,
      stack: [],
      defs: defs,
      function: nil,
      line: line,
      version: 0
    }

    public = for {def, :def} <- exports, do: def
    {quoted, state} = Enum.map_reduce(public, state, &compile_each/2)

    to_delete =
      for {def, stored} when stored != false <- state.defs do
        quote do
          Nx.Defn.Module.delete_definition(__MODULE__, unquote(def))
        end
      end

    {:__block__, [], to_delete ++ quoted}
  end

  defp compile_each({name, _arity} = def, state) do
    {{_kind, _meta, args, ast}, state} = get_cached_definition(def, state)

    quoted =
      quote do
        def unquote(name)(unquote_splicing(args)), do: unquote(ast)
      end

    {quoted, state}
  end

  defp get_cached_definition(def, state) do
    case state.defs do
      %{^def => false} -> get_and_cache_definition(def, state)
      %{^def => stored} -> {stored, state}
      %{} -> :none
    end
  end

  defp get_and_cache_definition(def, state) do
    {:v1, kind, meta, clauses} = Nx.Defn.Module.get_definition(state.module, def)

    with_def(meta, def, def, state, fn state ->
      case clauses do
        [] ->
          compile_error!(meta, state, "cannot have #{kind}n without clauses")

        [{meta, args, [], ast}] ->
          {args, state} = normalize_list(args, state)
          {ast, state} = normalize_block(ast, meta, state)
          {ast, state} = inline_locals(ast, state)
          result = {kind, meta, args, ast}
          state = put_in(state.defs[def], result)
          {result, state}

        [_, _ | _] ->
          compile_error!(meta, state, "cannot compile #{kind}n with multiple clauses")
      end
    end)
  end

  defp with_def(meta, def, call, state, fun) do
    %{function: previous_def, stack: previous_stack, line: previous_line} = state
    line = meta[:line] || previous_line
    {result, state} = fun.(%{state | function: def, stack: [call | previous_stack], line: line})
    {result, %{state | function: previous_def, stack: previous_stack, line: previous_line}}
  end

  ## Normalization

  defp normalize({:__block__, meta, _} = block, state) do
    normalize_block(block, meta, state)
  end

  defp normalize({:=, meta, [_, _] = args}, state) do
    {args, state} = normalize_list(args, state)
    {{:=, meta, args}, state}
  end

  defp normalize({name, meta, args}, state) when is_atom(name) and is_list(args) do
    {args, state} = normalize_list(args, state)
    {{:__local__, meta, [name | args]}, state}
  end

  defp normalize({name, meta, ctx}, state) when is_atom(name) and is_atom(ctx) do
    {version, meta} = Keyword.pop!(meta, :version)
    state = update_in(state.version, &max(&1, version))
    {{name, [counter: version] ++ meta, ctx}, state}
  end

  @allowed_nx_functions [add: 2, divide: 2, sum: 1, exp: 1]

  defp normalize({{:., _, [Nx, name]} = call, meta, args}, state) do
    arity = length(args)

    if {name, arity} not in @allowed_nx_functions do
      compile_error!(meta, state, "Nx.#{name}/#{arity} is not allowed inside defn")
    end

    {args, state} = normalize_list(args, state)
    {{call, meta, args}, state}
  end

  defp normalize(integer, state) when is_integer(integer) do
    {integer, state}
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

  defp normalize_block([nil], acc, meta, state) do
    compile_warn(meta, state, "body has nil return type, -1 will be returned instead")
    {Enum.reverse([-1 | acc]), state}
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

  ## Local inlining

  defp inline_locals({:__local__, meta, [name | args]}, state) do
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

  defp inline_locals({name, meta, args}, state) do
    {args, state} = inline_locals(args, state)
    {{name, meta, args}, state}
  end

  defp inline_locals({left, right}, state) do
    {left, state} = inline_locals(left, state)
    {right, state} = inline_locals(right, state)
    {{left, right}, state}
  end

  defp inline_locals(list, state) when is_list(list) do
    Enum.map_reduce(list, state, &inline_locals/2)
  end

  defp inline_locals(other, state) do
    {other, state}
  end

  ## Shared helpers

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
