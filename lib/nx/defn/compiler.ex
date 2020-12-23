defmodule Nx.Defn.Compiler do
  @moduledoc """
  The specification and helper functions for custom `defn` compilers.
  """

  @as_is_nx_functions [:tensor]
  @forbidden_nx_functions [:device_read, :device_deallocate, :device_transfer, :type]

  @doc """
  The callback required to be implemented for each compiler.

  It receives the function compilation environment `env`, the
  function `kind`, the variables, the expr function, and the
  compiler options.

  It must call `fun` with a list where all vars have been
  converted to `Nx.Defn.Expr` parameters via `Nx.Defn.Expr.parameter/2`.

  Note the number of variables is not the same as the function
  arity as argument patterns have already been matched. The
  current name and arity can be found under `env.function`.

  The callback uses double underscores so it can be defined
  at root modules without affecting the module's main API.
  """
  @callback __compile__(
              env :: Macro.Env.t(),
              kind :: :def | :defp,
              vars :: [term],
              fun :: ([Nx.Defn.Expr.t()] -> Nx.Defn.Expr.t()),
              opts :: keyword
            ) :: term

  defguardp is_var(var)
            when is_tuple(var) and tuple_size(var) == 3 and is_atom(elem(var, 0)) and
                   is_atom(elem(var, 2))

  defguardp is_underscore(var)
            when is_tuple(var) and tuple_size(var) == 3 and elem(var, 0) == :_ and
                   is_atom(elem(var, 2))

  @doc false
  def __remote__(module, function, args) do
    defn = defn_name(function)

    try do
      apply(module, defn, args)
    catch
      :error, :undef ->
        stack =
          case __STACKTRACE__ do
            [{^module, ^defn, args_or_arity, info} | stack] ->
              [{module, function, args_or_arity, info} | stack]

            stack ->
              stack
          end

        :erlang.raise(:error, :undef, stack)
    end
  end

  @doc false
  def __compile__(%Macro.Env{module: module, file: file, line: line}, exports) do
    {:module, Nx} = Code.ensure_compiled(Nx)

    state = %{
      module: module,
      file: file,
      line: line,
      function: nil,
      version: 0,
      exports: exports
    }

    quoted = Enum.map(exports, &compile_each(&1, state))
    {:__block__, [], quoted}
  end

  defp compile_each({{name, _arity} = def, def_meta}, state) do
    {{kind, _meta, args, ast}, state} = get_and_normalize_definition(def, state)
    vars = collect_vars(args)
    {def_module, def_opts} = def_meta.compiler
    defn_name = defn_name(name)

    quote line: state.line do
      Nx.Defn.Module.delete_definition(__MODULE__, unquote(def))

      def unquote(name)(unquote_splicing(args)) do
        unquote(def_module).__compile__(
          __ENV__,
          unquote(kind),
          unquote(vars),
          fn unquote(vars) ->
            Nx.Defn.Expr.to_result(unquote(defn_name)(unquote_splicing(args)))
          end,
          unquote(Macro.escape(def_opts))
        )
      end

      Kernel.unquote(kind)(unquote(defn_name)(unquote_splicing(args)), do: unquote(ast))
    end
  end

  defp collect_vars(args) do
    {_, vars} =
      Macro.prewalk(args, [], fn
        var, acc when is_var(var) and not is_underscore(var) ->
          {var, [var | acc]}

        node, acc ->
          {node, acc}
      end)

    vars
  end

  defp get_and_normalize_definition(def, state) do
    {:v1, kind, meta, clauses} = Nx.Defn.Module.get_definition(state.module, def)
    state = %{state | function: def, line: meta[:line] || state.line, version: 0}

    case clauses do
      [] ->
        compile_error!(meta, state, "cannot have #{kind}n without clauses")

      [{meta, args, [], ast}] ->
        {args, state} = normalize_args(args, meta, state)
        {ast, state} = normalize(ast, state)
        assert_uniq_vars!(args, state)
        clause = {kind, [max_counter: state.version] ++ meta, args, ast}
        {clause, state}

      [_, _ | _] ->
        compile_error!(meta, state, "cannot compile #{kind}n with multiple clauses")
    end
  end

  ## Normalization

  defp normalize({special_form, meta, args}, state)
       when special_form in [:{}, :%{}, :__block__] do
    {args, state} = normalize_list(args, state)
    {{special_form, meta, args}, state}
  end

  defp normalize({:=, meta, [left, right]}, state) do
    {left, state} = normalize(left, state)
    assert_uniq_vars!(left, state)
    {right, state} = normalize(right, state)
    {{:=, meta, [left, right]}, state}
  end

  defp normalize({name, meta, args} = expr, state) when is_atom(name) and is_list(args) do
    pair = {name, length(args)}

    case state.exports do
      %{^pair => _} ->
        {args, state} = normalize_list(args, state)
        {{defn_name(name), meta, args}, state}

      %{} ->
        invalid_numerical_expression!(expr, state)
    end
  end

  defp normalize(underscore, state) when is_underscore(underscore) do
    {underscore, state}
  end

  defp normalize({name, meta, ctx} = var, state) when is_var(var) do
    {version, meta} = Keyword.pop!(meta, :version)
    state = update_in(state.version, &max(&1, version))
    {{name, [counter: version, generated: true] ++ meta, ctx}, state}
  end

  defp normalize({{:., _, [Nx, name]} = call, meta, args}, state)
      when name in @as_is_nx_functions do
    {args, state} = normalize_list(args, state)
    {{call, meta, args}, state}
  end

  defp normalize({{:., _, [Nx, name]}, meta, args}, state)
      when name in @forbidden_nx_functions do
    arity = length(args)
    compile_error!(meta, state, "Nx.#{name}/#{arity} is not allowed inside defn")
  end

  defp normalize({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    arity = length(args)

    unless function_exported?(Nx, name, arity) do
      compile_error!(meta, state, "undefined function Nx.#{name}/#{arity}")
    end

    {args, state} = normalize_list(args, state)
    {{{:., dot_meta, [Nx.Defn.Expr, name]}, meta, args}, state}
  end

  defp normalize({{:., _, [Nx.Defn.Kernel, :transform]} = call, meta, [module, ast, opts]}, state) do
    unless is_atom(module) do
      compile_error!(
        meta,
        state,
        "expected the first argument of Nx.Defn.Kernel.transform/3 to be a module, " <>
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
    {{call, meta, [module, ast, opts]}, state}
  end

  defp normalize({{:., dot_meta, [remote, name]}, meta, args}, state)
       when is_atom(remote) and is_atom(name) do
    {args, state} = normalize_list(args, state)
    {{{:., dot_meta, [__MODULE__, :__remote__]}, meta, [remote, name, args]}, state}
  end

  defp normalize({left, right}, state) do
    {left, state} = normalize(left, state)
    {right, state} = normalize(right, state)
    {{left, right}, state}
  end

  defp normalize(list, state) when is_list(list) do
    normalize_list(list, state)
  end

  defp normalize(literal, state)
       when is_number(literal) or is_atom(literal) or is_bitstring(literal) do
    {literal, state}
  end

  defp normalize(expr, state) do
    invalid_numerical_expression!(expr, state)
  end

  defp normalize_list(list, state) do
    Enum.map_reduce(list, state, &normalize/2)
  end

  defp invalid_numerical_expression!(expr, state) do
    compile_error!(
      maybe_meta(expr),
      state,
      "invalid numerical expression: #{Macro.to_string(expr)}"
    )
  end

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

  ## Shared helpers

  defp maybe_meta({_, meta, _}), do: meta
  defp maybe_meta(_), do: []

  defp assert_uniq_vars!(ast, state) do
    Macro.prewalk(ast, %{}, fn
      var, acc when is_var(var) and not is_underscore(var) ->
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

  defp compile_error!(meta, state, description) do
    line = meta[:line] || state.line
    raise CompileError, line: line, file: state.file, description: description
  end

  defp defn_name(name), do: :"__defn:#{name}__"
end
