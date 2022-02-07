defmodule Nx.Defn.Compiler do
  @moduledoc """
  The specification and helper functions for custom `defn` compilers.
  """

  @type expr :: Nx.t() | tuple() | %{optional(term()) => expr()}

  @doc """
  Callback for JIT compilation.

  It receives an opaque `key` used for caching, the function
  `vars`, the function which builds an expression, and the compiler
  options.

  It must call `fun` with the vars as a list of arguments.
  Note the `key` does not include the `vars` in its cache.
  Therefore, if you want to cache the result of `fun.(vars)`,
  you likely want to include the vars in the cache key.
  Given `vars` are all tensors, it is often a matter of
  retrieving its type, shape, and names.

  The callback uses double underscores so it can be defined
  at root modules without affecting the module's main API.
  """
  @callback __jit__(
              key :: term,
              vars :: [Nx.t()],
              ([Nx.t()] -> expr()),
              opts :: keyword
            ) :: expr

  @doc """
  Callback for streaming (on top of JIT compilation).

  It receives the same arguments as `c:__jit__/4` with the addition
  of the streaming and accumulator templates. It must return a struct
  that implements the `Nx.Stream` protocol.
  """
  @callback __stream__(
              key :: term,
              stream,
              acc,
              vars :: [Nx.t()],
              ([Nx.t()] -> acc),
              opts :: keyword
            ) :: Nx.Stream.t()
            when stream: expr(), acc: expr()

  # Modules allowed in defn
  @allowed_modules [Nx, Nx.Constants, Nx.Defn, Nx.Defn.Kernel, Nx.LinAlg]

  # These operations do not have valid meaning for Nx.Defn.Expr
  @forbidden_ops [:backend_copy, :backend_deallocate, :backend_transfer] ++
                   [:to_binary, :to_number, :to_flat_list, :to_heatmap, :to_batched_list] ++
                   [:from_numpy, :from_numpy_archive, :compatible?, :default_backend]

  defguardp is_var(var)
            when is_tuple(var) and tuple_size(var) == 3 and is_atom(elem(var, 0)) and
                   is_atom(elem(var, 2))

  defguardp is_underscore(var)
            when is_tuple(var) and tuple_size(var) == 3 and elem(var, 0) == :_ and
                   is_atom(elem(var, 2))

  @doc """
  Returns the current compiler.

  Returns nil if we are not inside `defn`.
  """
  def current() do
    Process.get(Nx.Defn.Compiler)
  end

  ## JIT/Stream

  @doc false
  def __jit__(fun, args, opts) do
    {compiler, tail} = runtime(fun, args, opts)
    Kernel.apply(compiler, :__jit__, [fun | tail])
  end

  @doc false
  def __stream__(fun, input, acc, args, opts) do
    {compiler, tail} = runtime(fun, [input, acc | args], opts)
    Kernel.apply(compiler, :__stream__, [fun, input, acc | tail])
  end

  defp runtime(fun, args, opts) do
    {compiler, opts} = Keyword.pop(opts, :compiler, Nx.Defn.Evaluator)
    tensors = Nx.Defn.Composite.from_runtime_args(args, [])
    runtime_fun = &runtime_fun(&1, fun, args, compiler)
    {compiler, [tensors, runtime_fun, opts]}
  end

  defp runtime_fun(tensors, fun, args, compiler) do
    tuple = Nx.default_backend()
    Nx.default_backend(Nx.Defn.Expr)
    Process.put(Nx.Defn.Compiler, compiler)

    try do
      args = Nx.Defn.Composite.args_to_params(args, tensors)

      fun
      |> apply(args)
      |> Nx.Defn.Composite.to_result()
    after
      Nx.default_backend(tuple)
      Process.delete(Nx.Defn.Compiler)
    end
  end

  ## Compiler

  @doc false
  def __remote__(module, function, defn, args) do
    try do
      apply(module, defn, args)
    catch
      :error, :undef ->
        stack =
          case __STACKTRACE__ do
            [{^module, ^defn, args_or_arity, info}, _ | stack] ->
              if function_exported?(module, function, length(args)) do
                formatted = Exception.format_mfa(module, function, length(args))

                reraise "cannot invoke #{formatted} inside defn because it was not defined with defn",
                        stack
              else
                [{module, function, args_or_arity, info} | stack]
              end

            stack ->
              stack
          end

        :erlang.raise(:error, :undef, stack)
    end
  end

  @doc false
  def __compile__(%Macro.Env{module: module, file: file, line: line}, exports) do
    defns =
      for {{name, arity}, %{defaults: defaults}} <- exports,
          arity <- (arity - map_size(defaults))..arity,
          do: {name, arity}

    state = %{
      module: module,
      file: file,
      line: line,
      function: nil,
      defns: MapSet.new(defns),
      rewrite_underscore?: false
    }

    quoted = Enum.map(exports, &compile_each(&1, state))
    {:__block__, [], quoted}
  end

  defp compile_each({{name, arity} = def, def_meta}, state) do
    %{defaults: defaults} = def_meta
    {{kind, _meta, args, ast}, state} = get_and_normalize_definition(def, state)

    defn_name = defn_name(name)

    defn_args =
      Enum.with_index(args, fn arg, i ->
        case defaults do
          %{^i => {meta, default}} -> {:\\, meta, [arg, default]}
          %{} -> arg
        end
      end)

    all_args = Macro.generate_arguments(arity, __MODULE__)

    fn_args =
      for {arg, i} <- Enum.with_index(all_args),
          not Map.has_key?(defaults, i),
          do: arg

    fun =
      if defaults == [] do
        quote do
          &(unquote(Macro.var(defn_name, __MODULE__)) / unquote(arity))
        end
      else
        quote do
          fn unquote_splicing(fn_args) -> unquote(defn_name)(unquote_splicing(all_args)) end
        end
      end

    quote line: state.line do
      Module.delete_definition(__MODULE__, unquote(def))

      Kernel.unquote(kind)(unquote(name)(unquote_splicing(all_args))) do
        if Process.get(Nx.Defn.Compiler) do
          unquote(defn_name)(unquote_splicing(all_args))
        else
          Nx.Defn.Compiler.__runtime__(unquote(fun), unquote(fn_args))
        end
      end

      Kernel.unquote(kind)(unquote(defn_name)(unquote_splicing(defn_args)), do: unquote(ast))
    end
  end

  @doc false
  def __runtime__(fun, args) do
    {compiler, compiler_opts} =
      Keyword.pop(Nx.Defn.default_options(), :compiler, Nx.Defn.Evaluator)

    {cache, tensors} = Nx.Defn.Composite.from_compile_args(args, fun)

    compiler.__jit__(
      cache,
      Nx.Defn.Composite.from_runtime_args(tensors, []),
      &runtime_fun(&1, fun, args, compiler),
      compiler_opts
    )
  end

  defp get_and_normalize_definition(def, state) do
    {:v1, kind, meta, clauses} = Module.get_definition(state.module, def)
    state = %{state | function: def, line: meta[:line] || state.line, rewrite_underscore?: true}

    case clauses do
      [] ->
        compile_error!(meta, state, "cannot have #{kind}n without clauses")

      [{meta, args, [], ast}] ->
        {args, state} = normalize_args(args, meta, state)
        {ast, state} = normalize(ast, %{state | rewrite_underscore?: false})
        {{kind, meta, args, ast}, state}

      [_, _ | _] ->
        compile_error!(meta, state, "cannot compile #{kind}n with multiple clauses")
    end
  end

  ## Normalization

  defp normalize({:%, meta, [aliases, {:%{}, map_meta, [{:|, update_meta, [map, args]}]}]}, state) do
    {map, state} = normalize(map, state)
    {args, state} = normalize(args, state)
    {{:%, meta, [aliases, {:%{}, map_meta, [{:|, update_meta, [map, args]}]}]}, state}
  end

  defp normalize({:%, meta, [aliases, {:%{}, map_meta, args}]}, state) do
    {args, state} = normalize(args, state)
    {{:%, meta, [aliases, {:%{}, map_meta, args}]}, state}
  end

  defp normalize({:%{}, meta, [{:|, update_meta, [map, args]}]}, state) do
    {map, state} = normalize(map, state)
    {args, state} = normalize(args, state)
    {{:%{}, meta, [{:|, update_meta, [map, args]}]}, state}
  end

  defp normalize({special_form, meta, args}, state)
       when special_form in [:{}, :%{}, :%, :__block__] do
    {args, state} = normalize_list(args, state)
    {{special_form, meta, args}, state}
  end

  defp normalize({:=, meta, [left, right]}, state) do
    {left, state} = normalize(left, state)
    assert_uniq_vars!(left, state)
    {right, state} = normalize(right, state)
    {{:=, meta, [left, right]}, state}
  end

  defp normalize({:&, _, _} = expr, state) do
    {expr, state}
  end

  defp normalize({:fn, meta, clauses}, state) do
    unless match?([_], clauses) do
      compile_error!(meta, state, "only a single clause is allowed inside fn")
    end

    {clauses, state} =
      Enum.map_reduce(clauses, state, fn {:->, clause_meta, [args, body]}, state ->
        {args, state} = normalize_args(args, meta, state)
        {body, state} = normalize(body, state)
        {{:->, clause_meta, [args, body]}, state}
      end)

    {{:fn, meta, clauses}, state}
  end

  defp normalize({:cond, meta, [[do: clauses]]}, state) do
    {[{last_meta, {last_condition, last_expr}} | rest], state} =
      Enum.reduce(clauses, {[], state}, fn {:->, meta, [[condition], expr]}, {acc, state} ->
        {condition, state} = normalize(condition, state)
        {expr, state} = normalize(expr, state)
        {[{meta, {condition, expr}} | acc], state}
      end)

    if rest == [] do
      compile_error!(meta, state, "cond must have at least 2 clauses, got 1")
    end

    if not is_atom(last_condition) or last_condition == nil or last_condition == false do
      compile_error!(
        last_meta,
        state,
        "expected the last clause of cond to match on an atom, " <>
          "such as true or :otherwise, got: #{Macro.to_string(last_condition)}"
      )
    end

    ast =
      quote do
        Nx.Defn.Expr.defn_cond(
          unquote(state.file),
          unquote(Enum.reverse(rest)),
          unquote(last_expr)
        )
      end

    {ast, state}
  end

  defp normalize({name, meta, args} = expr, state) when is_atom(name) and is_list(args) do
    pair = {name, length(args)}

    if pair in state.defns do
      {args, state} = normalize_list(args, state)
      {{defn_name(name), meta, args}, state}
    else
      invalid_numerical_expression!(expr, state)
    end
  end

  defp normalize(underscore, state) when is_underscore(underscore) do
    {underscore, state}
  end

  defp normalize(var, state) when is_var(var) do
    {normalize_var(var), state}
  end

  defp normalize({{:., dot_meta, [fun]}, meta, args}, state) do
    {fun, state} = normalize(fun, state)
    {args, state} = normalize_list(args, state)
    {{{:., dot_meta, [fun]}, meta, args}, state}
  end

  defp normalize({{:., _, [Nx.Defn.Kernel, :transform]} = call, meta, [ast, fun]}, state) do
    {ast, state} = normalize(ast, state)

    fun =
      Macro.prewalk(fun, fn
        var when is_var(var) -> normalize_var(var)
        node -> node
      end)

    {{call, meta, [ast, fun]}, state}
  end

  defp normalize({{:., _, [Nx.Defn.Kernel, :hook]} = call, meta, [ast | rest]}, state) do
    {ast, state} = normalize(ast, state)
    {{call, meta, [ast | rest]}, state}
  end

  defp normalize(
         {{:., _, [Nx.Defn.Kernel, :hook_token]} = call, meta, [token, ast | rest]},
         state
       ) do
    {token, state} = normalize(token, state)
    {ast, state} = normalize(ast, state)
    {{call, meta, [token, ast | rest]}, state}
  end

  defp normalize({{:., dot_meta, [mod, name]}, meta, args}, state) when mod in @allowed_modules do
    if name in @forbidden_ops do
      mfa = Exception.format_mfa(mod, name, length(args))
      compile_error!(meta, state, "#{mfa} is not allowed inside defn")
    end

    {args, state} = normalize_list(args, state)
    {{{:., dot_meta, [mod, name]}, meta, args}, state}
  end

  defp normalize({{:., _, [Access, :get]} = call, meta, args}, state) do
    {args, state} = normalize_list(args, state)
    {{call, meta, args}, state}
  end

  defp normalize({{:., dot_meta, [remote, name]}, meta, args}, state)
       when is_atom(remote) and is_atom(name) do
    {args, state} = normalize_list(args, state)

    {{{:., dot_meta, [__MODULE__, :__remote__]}, meta, [remote, name, defn_name(name), args]},
     state}
  end

  defp normalize({{:., dot_meta, [remote, name]}, meta, []}, state) when is_atom(name) do
    {remote, state} = normalize(remote, state)
    {{{:., dot_meta, [Map, :fetch!]}, meta, [remote, name]}, state}
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
       when is_number(literal) or is_atom(literal) or is_binary(literal) do
    {literal, state}
  end

  defp normalize(expr, state) do
    invalid_numerical_expression!(expr, state)
  end

  defp normalize_var({name, meta, ctx} = var) do
    case Keyword.pop(meta, :version) do
      {nil, _} -> var
      {version, meta} -> {name, [counter: version, generated: true] ++ meta, ctx}
    end
  end

  defp normalize_list(list, state) do
    Enum.map_reduce(list, state, &normalize/2)
  end

  defp invalid_numerical_expression!(expr, state) do
    string = expr |> Macro.to_string() |> String.replace("\n", "\n    ")

    compile_error!(
      maybe_meta(expr),
      state,
      "invalid numerical expression:\n\n    #{string}\n"
    )
  end

  ## Normalize args

  defp normalize_args(args, meta, state) when is_list(args) do
    {args, state} = Enum.map_reduce(args, state, &normalize_arg(&1, meta, &2))
    assert_uniq_vars!(args, state)
    {args, state}
  end

  defp normalize_arg(var, _meta, state) when is_var(var) do
    if state.rewrite_underscore? and is_underscore(var) do
      {Macro.unique_var(:arg, state.module), state}
    else
      normalize(var, state)
    end
  end

  defp normalize_arg({:%, meta, [aliases, {:%{}, meta, args}]}, _meta, state) do
    {args, state} =
      Enum.map_reduce(args, state, fn {k, v}, acc ->
        {v, acc} = normalize_arg(v, meta, acc)
        {{k, v}, acc}
      end)

    {{:%, meta, [aliases, {:%{}, meta, args}]}, state}
  end

  defp normalize_arg({:%{}, meta, args}, _meta, state) do
    {args, state} =
      Enum.map_reduce(args, state, fn {k, v}, acc ->
        {v, acc} = normalize_arg(v, meta, acc)
        {{k, v}, acc}
      end)

    {{:%{}, meta, args}, state}
  end

  defp normalize_arg({op, meta, args}, _meta, state) when op in [:{}, :=] do
    {args, state} = Enum.map_reduce(args, state, &normalize_arg(&1, meta, &2))
    {{op, meta, args}, state}
  end

  defp normalize_arg({left, right}, meta, state) do
    {left, state} = normalize_arg(left, meta, state)
    {right, state} = normalize_arg(right, meta, state)
    {{:{}, meta, [left, right]}, state}
  end

  defp normalize_arg(expr, meta, state) do
    compile_error!(
      meta,
      state,
      "only variables, tuples, maps, and structs are allowed as patterns in defn, got: #{Macro.to_string(expr)}"
    )
  end

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

  ## Helpers

  defp maybe_meta({_, meta, _}), do: meta
  defp maybe_meta(_), do: []

  defp compile_error!(meta, state, description) do
    line = meta[:line] || state.line
    raise CompileError, line: line, file: state.file, description: description
  end

  defp defn_name(name), do: :"__defn:#{name}__"
end
