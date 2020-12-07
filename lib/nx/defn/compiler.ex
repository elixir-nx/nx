defmodule Nx.Defn.Compiler do
  @moduledoc """
  The specification and helper functions for custom `defn` compilers.
  """

  @forbidden_nx_functions [tensor: 1, tensor: 2, device_read: 1, device_deallocate: 1] ++
                            [device_transfer: 1, device_transfer: 2, device_transfer: 3]

  @known_keywords [:type, :axis]

  @doc """
  The callback required to be implemented for each compiler.

  It receives the module compilation environment `env`, the
  function `kind`, the function `metadata`, the variables,
  the AST, and the compiler options. The AST is guaranteed
  to be valid Elixir code. It must return valid Elixir AST.

  Note the number of variables is not the same as the function
  arity as argument patterns have already been matched. The
  current name and arity can be found under `env.function`.

  Each variable in the AST has a `:counter` metadata which is
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
              env :: Macro.Env.t(),
              kind :: :def | :defp,
              metadata :: keyword,
              vars :: [Macro.t()],
              ast :: Macro.t(),
              opts :: keyword
            ) :: Macro.t()

  defguardp is_var(var) when is_atom(elem(var, 0)) and is_atom(elem(var, 2))

  # The compiler has four passes.
  #
  # 1. Normalize and validate the AST written by the user.
  #
  # 2. Expand patterns and inline all local functions. This is
  #    the result stored in each module for cross-module inlining.
  #
  # 3. Inline remote calls (and then transforms + remote calls recursively).
  #
  # 4. Invoke the compiler chosen by the user.
  #
  # The user compiler may have multiple passes and it has the ability
  # to run said additional passes either at runtime or compilation time.
  @doc false
  def compile(%Macro.Env{module: module, file: file, line: line} = env, exports) do
    {:module, Nx} = Code.ensure_compiled(Nx)
    cache = for {def, _meta} <- exports, into: %{}, do: {def, false}
    public = for {def, %{kind: :def} = meta} <- exports, do: {def, meta}

    state = %{
      module: module,
      file: file,
      stack: [],
      cache: cache,
      function: nil,
      line: line,
      version: 0,
      remotes: %{}
    }

    {quoted, state} = Enum.map_reduce(public, state, &compile_each(env, &1, &2))

    remotes =
      for {remote, _} <- state.remotes do
        quote do
          require unquote(remote)
        end
      end

    to_delete =
      for {def, stored} when stored != false <- state.cache do
        quote do
          Nx.Defn.Module.delete_definition(__MODULE__, unquote(def))
        end
      end

    catch_all =
      quote do
        def __defn__(_, _), do: nil
      end

    {:__block__, [], remotes ++ to_delete ++ quoted ++ [catch_all]}
  end

  defp compile_each(env, {{name, arity} = def, def_meta}, state) do
    {{kind, meta, args, ast}, state} = get_cached_definition(def, state)
    defn_ast = Macro.escape({meta, args, ast})

    # Inline remotes and transforms
    env = %{env | function: def, line: meta[:line] || env.line}
    {ast, state} = inline_remote(ast, state)
    {ast, state} = inline_transforms(env, ast, state)

    # Now invoke the compiler
    {def_module, def_opts} = def_meta.compiler
    meta = Keyword.put(meta, :max_counter, state.version)
    compiled_body = def_module.__compile__(env, kind, meta, collect_vars(args), ast, def_opts)

    quoted =
      quote do
        def unquote(name)(unquote_splicing(args)), do: unquote(compiled_body)
        def __defn__(unquote(name), unquote(arity)), do: unquote(defn_ast)
      end

    {quoted, state}
  end

  defp collect_vars(args) do
    {_, vars} =
      Macro.prewalk(args, [], fn
        var, acc when is_var(var) ->
          {var, [var | acc]}

        node, acc ->
          {node, acc}
      end)

    vars
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
          {args, state} = normalize_args(args, meta, state)
          assert_uniq_vars!(args, state)
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

  defp normalize({:=, meta, [left, right]}, state) do
    {left, state} = normalize(left, state)
    assert_uniq_vars!(left, state)
    {right, state} = normalize(right, state)
    {{:=, meta, [left, right]}, state}
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
       when is_atom(name) and is_list(args) and name not in [:%, :%{}, :^, :<<>>] do
    {args, state} = normalize_list(args, state)
    {{:__local__, meta, [name | args]}, state}
  end

  defp normalize({name, meta, ctx} = var, state) when is_var(var) do
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
    args = rewrite_nx_args(name, args)
    {{call, meta, args}, state}
  end

  defp normalize({{:., _, [Nx.Defn.Kernel, :transform]}, meta, [module, ast, opts]}, state) do
    unless is_atom(module) do
      compile_error!(
        meta,
        state,
        "expected the first argument of Nx.Defn.Kernel.transform/3 to a module, " <>
          "got: #{inspect(module)}"
      )
    end

    unless Keyword.keyword?(opts) and Enum.all?(opts, fn {_, v} -> is_atom(v) or is_number(v) end) do
      compile_error!(
        meta,
        state,
        "expected the second argument of Nx.Defn.Kernel.transform/3 to a keyword list with " <>
          "atoms and numbers as values, got: #{inspect(opts)}"
      )
    end

    {ast, state} = normalize(ast, state)
    {{:__transform__, meta, [module, ast, opts]}, state}
  end

  defp normalize({{:., _, [remote, name]}, meta, args}, state)
       when is_atom(remote) and is_atom(name) do
    {args, state} = normalize_list(args, state)
    {{:__remote__, meta, [remote, name | args]}, state}
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

  ## Normalize nx calls

  defp rewrite_nx_args(:sum, [arg]), do: [arg, []]
  defp rewrite_nx_args(:random_uniform, [arg]), do: [arg, 0.0, 1.0, []]
  defp rewrite_nx_args(:random_uniform, [arg, opts]), do: [arg, 0.0, 1.0, opts]
  defp rewrite_nx_args(:random_uniform, [arg, min, max]), do: [arg, min, max, []]
  defp rewrite_nx_args(:random_normal, [arg]), do: [arg, 0.0, 1.0, []]
  defp rewrite_nx_args(:random_normal, [arg, opts]), do: [arg, 0.0, 1.0, opts]
  defp rewrite_nx_args(:random_normal, [arg, min, max]), do: [arg, min, max, []]
  defp rewrite_nx_args(_, args), do: args

  ## Normalize args

  defp normalize_args(args, meta, state) when is_list(args) do
    Enum.map_reduce(args, state, &normalize_args(&1, meta, &2))
  end

  defp normalize_args(var, _meta, state) when is_var(var) do
    normalize(var, state)
  end

  defp normalize_args({:{}, meta, args}, _meta, state) do
    {args, state} = normalize_args(args, meta, state)
    {{:{}, meta, args}, state}
  end

  defp normalize_args({left, right}, meta, state) do
    {args, state} = normalize_args([left, right], meta, state)
    {{:{}, meta, args}, state}
  end

  defp normalize_args(expr, meta, state) do
    compile_error!(
      meta,
      state,
      "only variables and tuples are allowed as arguments in defn, got: #{Macro.to_string(expr)}"
    )
  end

  ## Normalize block

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

  ## Expansion and local inlining

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
    {args, state} = expand(args, state)
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
      {{_kind, _meta, patterns, ast}, state} ->
        {ast, version} = inline(meta, patterns, ast, args, state.version)
        {ast, %{state | version: version}}

      :none ->
        compile_error!(
          meta,
          state,
          "cannot invoke #{name}/#{arity} because it was not defined with defn"
        )
    end
  end

  defp expand({expr, meta, args}, state) do
    {expr, state} = expand(expr, state)
    {args, state} = expand(args, state)
    {{expr, meta, args}, state}
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

  defp expand_pattern(_meta, var, patterns, vars, expr, _state) when is_var(var) do
    {patterns, [var | vars], expr}
  end

  defp expand_pattern(meta, pattern, patterns, vars, expr, state) do
    pattern = validate_pattern!(meta, pattern, state)
    {[pattern | patterns], vars, expr}
  end

  defp validate_pattern!(meta, {left, right}, _state), do: {:{}, meta, [left, right]}
  defp validate_pattern!(_meta, {:{}, _, _} = tuple, _state), do: tuple
  defp validate_pattern!(_meta, var, _state) when is_var(var), do: var

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

  defp expand_nested_patterns([var | args], meta, acc, nested, state) when is_var(var) do
    expand_nested_patterns(args, meta, [var | acc], nested, state)
  end

  defp expand_nested_patterns([expr | args], meta, acc, nested, state) do
    {var, state} = new_var(state)
    expand_nested_patterns(args, meta, [var | acc], [{:=, meta, [expr, var]} | nested], state)
  end

  defp expand_nested_patterns([], _meta, acc, nested, state) do
    {Enum.reverse(acc), nested, state}
  end

  ### Inline

  defp inline(meta, patterns, ast, args, version) do
    version = version + 1

    {{patterns, ast}, max_version} =
      Macro.prewalk({patterns, ast}, version, fn
        {var, meta, ctx}, max_version when is_atom(var) and is_atom(ctx) ->
          {var_version, meta} =
            Keyword.get_and_update!(meta, :counter, &{&1 + version, &1 + version})

          {{var, meta, ctx}, max(max_version, var_version)}

        node, max_version ->
          {node, max_version}
      end)

    case patterns do
      [] ->
        {ast, max_version}

      _ ->
        assigns =
          patterns
          |> Enum.zip(args)
          |> Enum.map(fn {var, arg} -> {:=, meta, [var, arg]} end)

        {{:__block__, meta, assigns ++ [ast]}, max_version}
    end
  end

  defp inline_remote(ast, state) do
    Macro.prewalk(ast, state, fn
      {:__remote__, meta, [module, name | args]}, state ->
        arity = length(args)

        if Code.ensure_compiled(module) != {:module, module} do
          compile_error!(
            meta,
            state,
            "cannot invoke #{inspect(module)}.#{name}/#{arity} because #{inspect(module)} " <>
              "does not exist or it is currently unavailable due to a deadlock (defn does " <>
              "not allow co-recursive definitions)"
          )
        end

        unless defn = function_exported?(module, :__defn__, 2) && module.__defn__(name, arity) do
          compile_error!(
            meta,
            state,
            "undefined numerical function #{inspect(module)}.#{name}/#{arity}"
          )
        end

        {_meta, patterns, ast} = defn
        {ast, version} = inline(meta, patterns, ast, args, state.version)
        state = put_in(state.remotes[module], true)
        {ast, %{state | version: version}}

      expr, state ->
        {expr, state}
    end)
  end

  ## Transforms

  # TODO: define proper behaviour
  defp inline_transforms(env, ast, state) do
    Macro.prewalk(ast, state, fn
      {:__transform__, meta, [module, ast, opts]}, state ->
        if Code.ensure_compiled(module) != {:module, module} do
          compile_error!(
            meta,
            state,
            "cannot invoke transform #{inspect(module)} because it is not defined"
          )
        end

        {version, ast} = module.__transform__(env, state.version, meta, ast, opts)
        state = put_in(state.remotes[module], true)
        inline_remote(ast, %{state | version: version})

      expr, state ->
        {expr, state}
    end)
  end

  ## Shared helpers

  defp new_var(state) do
    counter = state.version + 1
    {{:nvar, [counter: counter], __MODULE__}, %{state | version: counter}}
  end

  defp maybe_meta({_, meta, _}), do: meta
  defp maybe_meta(_), do: []

  defp assert_uniq_vars!(ast, state) do
    Macro.prewalk(ast, %{}, fn
      var, acc when is_var(var) ->
        meta = elem(var, 1)
        counter = Keyword.fetch!(meta, :counter)

        case acc do
          %{^counter => var} ->
            compile_error!(
              meta,
              state,
              "variable \"#{Macro.to_string(var)}\" appears twice in pattern " <>
                Macro.to_string(ast)
            )

          %{} ->
            {var, Map.put(acc, counter, var)}
        end

      node, acc ->
        {node, acc}
    end)

    :ok
  end

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
