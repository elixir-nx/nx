defmodule Nx.Defn.Compiler do
  @moduledoc """
  The specification and helper functions for custom `defn` compilers.
  """

  @doc """
  Callback for compilation.

  It receives an opaque `key` used for caching, the function
  `vars`, the function `fun` which builds a defn expression,
  a list of argument list in `args_list`, and the compiler options.

  It must call `fun` with the `vars` as arguments. Note the `key`
  does not include the `vars` in its cache. Therefore, if you want
  to cache the result of `fun.(vars)`, you likely want to include
  the vars in the cache key. `vars` is a list of containers expressions.

  Once the expression is built and compiled, it must be invoked
  for each list of arguments in `args_list`. In a nutshell, `vars`
  are used to build the expression from `fun` which is then
  invoked for each list of arguments in `args_list`. All lists
  in `args_list` are guaranteed to be flat lists of the same length,
  containing zero-arity functions that return tensors of the same type,
  shape, and name.

  The callback uses double underscores so it can be defined
  at root modules without affecting the module's main API.
  """
  @callback __jit__(
              key :: term,
              vars :: [Nx.Container.t()],
              fun :: ([Nx.Container.t()] -> Nx.Container.t()),
              args_list :: [[(() -> Nx.t())]],
              opts :: keyword
            ) :: [Nx.Container.t()]

  @doc """
  Callback for compilation.

  It receives an opaque `key` used for caching, the function
  `vars`, the function `fun` which builds a defn expression,
  and the compiler options. It must call `fun` with the `vars`
  as arguments.

  It returns a function that receives a list of arguments and
  returns a list of results.

  The callback uses double underscores so it can be defined
  at root modules without affecting the module's main API.
  """
  @callback __compile__(
              key :: term,
              vars :: [Nx.Container.t()],
              fun :: ([Nx.Container.t()] -> Nx.Container.t()),
              opts :: keyword
            ) :: ([[Nx.t()]] -> [Nx.Container.t()])

  @doc """
  Callback for streaming (on top of JIT compilation).

  It receives the same arguments as `c:__jit__/5` with the addition
  of the streaming input and accumulator templates. If the input
  and accumulator are containers, they are kept in their container
  shapes. As in `c:__jit__/5`, both `vars` and `args_list` are flat
  lists of tensors (without their container shape).

  It must return a struct that implements the `Nx.Stream` protocol.
  """
  @callback __stream__(
              key :: term,
              input,
              acc,
              vars :: [Nx.t()],
              fun :: ([Nx.t()] -> {output, acc}),
              args_list :: [[(() -> Nx.t())]],
              opts :: keyword
            ) :: [Nx.Stream.t()]
            when input: Nx.Container.t(), output: Nx.Container.t(), acc: Nx.Container.t()

  # Modules allowed in defn
  @allowed_modules [Nx, Nx.Constants, Nx.Defn, Nx.Defn.Kernel, Nx.LinAlg, Nx.Type]

  # These operations do not have valid meaning for Nx.Defn.Expr
  @forbidden_ops [:backend_copy, :backend_deallocate, :backend_transfer] ++
                   [:to_binary, :to_number, :to_flat_list, :to_heatmap, :to_batched] ++
                   [:from_numpy, :from_numpy_archive, :compatible?, :default_backend] ++
                   [:serialize, :deserialize]

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
  def __compile__(fun, params, opts) do
    {compiler, runtime_fun, opts} = prepare_options(fun, opts)
    compiler.__compile__(fun, params, runtime_fun, opts)
  end

  @doc false
  def __jit__(fun, params, args_list, opts) do
    {compiler, runtime_fun, opts} = prepare_options(fun, opts)
    compiler.__jit__(fun, params, runtime_fun, args_list, opts)
  end

  @doc false
  def __stream__(fun, input, acc, params, args_list, opts) do
    {compiler, runtime_fun, opts} = prepare_options(fun, opts)
    compiler.__stream__(fun, input, acc, params, runtime_fun, args_list, opts)
  end

  defp prepare_options(fun, opts) do
    {compiler, opts} = Keyword.pop(opts, :compiler, Nx.Defn.Evaluator)
    {compiler, &runtime_fun(&1, fun, compiler), opts}
  end

  defp runtime_fun(args, fun, compiler) do
    tuple = Nx.default_backend()
    Nx.default_backend(Nx.Defn.Expr)
    previous = Process.put(Nx.Defn.Compiler, compiler)

    try do
      fun
      |> apply(args)
      |> Nx.Defn.Composite.traverse(&Nx.Defn.Expr.tensor/1)
    after
      Nx.default_backend(tuple)

      if previous do
        Process.put(Nx.Defn.Compiler, compiler)
      else
        Process.delete(Nx.Defn.Compiler)
      end
    end
  end

  ## Compiler

  @doc false
  def __case__(arg) when is_tuple(arg) do
    arg |> Tuple.to_list() |> Enum.each(&__case__/1)
    arg
  end

  def __case__(arg) when is_number(arg) or is_atom(arg) do
    arg
  end

  def __case__(%_{} = arg) do
    arg
  end

  def __case__(arg) do
    raise ArgumentError, """
    only tuples, atoms, numbers, and structs are allowed as arguments to case/2 inside defn.
    Got: #{inspect(arg)}

    Consider using deftransform/2 or deftransformp/2 if you need to handle more complex cases
    """
  end

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

                message =
                  "cannot invoke #{formatted} inside defn because it was not defined with defn"

                detail =
                  case module do
                    IO ->
                      ". To print the runtime value of a tensor, use print_value/2. " <>
                        "To print the tensor expression, use print_expr/2"

                    _ ->
                      ""
                  end

                reraise message <> detail, stack
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
    {defn_exports, transform_exports} =
      Enum.split_with(exports, fn {_fun_arity, meta} -> meta.type == :numerical end)

    defns = compile_prepare_arities(defn_exports)
    transforms = compile_prepare_arities(transform_exports)

    state = %{
      module: module,
      file: file,
      line: line,
      function: nil,
      defns: defns,
      transforms: transforms,
      rewrite_underscore?: false
    }

    quoted =
      Enum.map(transform_exports, &compile_each_transform(&1, state)) ++
        Enum.map(defn_exports, &compile_each_defn(&1, state))

    {:__block__, [], quoted}
  end

  defp compile_prepare_arities(definitions) do
    for {{name, arity}, %{defaults: defaults}} <- definitions,
        arity <- (arity - map_size(defaults))..arity,
        into: MapSet.new(),
        do: {name, arity}
  end

  defp compile_each_defn({{name, arity} = def, def_meta}, state) do
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
    Module.delete_definition(state.module, def)

    entrypoint =
      quote line: state.line do
        Kernel.unquote(kind)(unquote(name)(unquote_splicing(all_args))) do
          if Process.get(Nx.Defn.Compiler) do
            unquote(defn_name)(unquote_splicing(all_args))
          else
            Nx.Defn.Compiler.__runtime__(
              &(unquote(Macro.var(defn_name, __MODULE__)) / unquote(arity)),
              unquote(all_args)
            )
          end
        end
      end

    impl =
      quote line: state.line do
        Kernel.unquote(kind)(unquote(defn_name)(unquote_splicing(defn_args)), do: unquote(ast))
      end

    {strip_definition_context(entrypoint), impl}
  end

  # If the definition has a context, we don't warn when it goes unused,
  # so we remove the context as we want to keep the original semantics.
  defp strip_definition_context({kind, meta, [signature, block]}) do
    {kind, meta, [Macro.update_meta(signature, &Keyword.delete(&1, :context)), block]}
  end

  defp compile_each_transform({{name, max_arity}, _def_meta}, state) do
    defn_name = defn_name(name)

    # {...} <- [Module...] is a trick so we can skip nil definitions for a given arity
    ast =
      for defn_arity <- 0..max_arity,
          {:v1, kind, meta, _clauses} <- [Module.get_definition(state.module, {name, defn_arity})] do
        defn_args = Macro.generate_arguments(defn_arity, __MODULE__)

        quote line: meta[:line] do
          Kernel.unquote(kind)(unquote(defn_name)(unquote_splicing(defn_args)),
            do: unquote(name)(unquote_splicing(defn_args))
          )
        end
      end

    {:__block__, [], ast}
  end

  @doc false
  def __runtime__(fun, args) do
    {compiler, compiler_opts} =
      Keyword.pop(Nx.Defn.default_options(), :compiler, Nx.Defn.Evaluator)

    {fun, params, flatten} = to_lazy_params(fun, args)
    runtime_fun = &runtime_fun(&1, fun, compiler)
    [res] = compiler.__jit__(fun, params, runtime_fun, [flatten], compiler_opts)
    res
  end

  defp get_and_normalize_definition(def, state) do
    {:v1, kind, meta, clauses} = Module.get_definition(state.module, def)
    state = %{state | function: def, line: meta[:line] || state.line, rewrite_underscore?: true}

    type_str = if kind == :def, do: "defn", else: "defnp"

    case clauses do
      [] ->
        compile_error!(meta, state, "cannot have #{type_str} without clauses")

      [{meta, args, [], ast}] ->
        {args, state} = normalize_args(args, meta, state)
        {ast, state} = normalize(ast, %{state | rewrite_underscore?: false})
        {{kind, meta, args, ast}, state}

      [_, _ | _] ->
        compile_error!(meta, state, "cannot compile #{type_str} with multiple clauses")
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

  defp normalize({:<<>>, meta, args}, state) do
    {args, state} =
      Enum.map_reduce(args, state, fn {:"::", meta, [left, right]}, acc ->
        {left, acc} =
          case left do
            {{:., _, [String.Chars, :to_string]} = dot, dot_meta, [left]} ->
              {left, acc} = normalize(left, acc)
              {{dot, dot_meta, [left]}, acc}

            _ ->
              normalize(left, acc)
          end

        {{:"::", meta, [left, right]}, acc}
      end)

    {{:<<>>, meta, args}, state}
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

  defp normalize({:case, meta, [expr, [do: clauses]]}, state) do
    {expr, state} = normalize(expr, state)

    {clauses, state} =
      Enum.map_reduce(clauses, state, fn {:->, clause_meta, [[head], body]}, state ->
        {when_meta, pattern, guard} =
          case head do
            {:when, when_meta, [pattern, guard]} -> {when_meta, pattern, guard}
            _ -> {clause_meta, head, true}
          end

        {pattern, vars} =
          Macro.postwalk(pattern, %{}, fn
            {:%{}, meta, [_ | _]} = map, _acc ->
              compile_error!(
                meta,
                state,
                "case/2 in defn does not allow matching on keys of maps and structs in patterns. Got: #{Macro.to_string(map)}"
              )

            {var, _meta, ctx} = triplet, acc when is_atom(var) and is_atom(ctx) ->
              {normalize_var(triplet), Map.put(acc, {var, ctx}, true)}

            other, acc ->
              {other, acc}
          end)

        guard =
          Macro.postwalk(guard, fn
            {var, meta, ctx} = triplet when is_atom(var) and is_atom(ctx) ->
              if is_map_key(vars, {var, ctx}) do
                normalize_var(triplet)
              else
                compile_error!(
                  meta,
                  state,
                  "case/2 in defn allow guards to only access variables defined in patterns. Got: #{var}"
                )
              end

            other ->
              other
          end)

        {body, state} = normalize(body, state)
        {{:->, clause_meta, [[{:when, when_meta, [pattern, guard]}], body]}, state}
      end)

    wrapped = {{:., meta, [__MODULE__, :__case__]}, meta, [expr]}
    {{:case, meta, [wrapped, [do: clauses]]}, state}
  end

  @cond_var_ast {:condition, [], __MODULE__}

  defp normalize({:cond, _meta, [[do: clauses]]}, state) do
    {clauses, state} =
      Enum.map_reduce(clauses, state, fn {:->, meta, [[condition], expr]}, state ->
        {condition, state} = normalize(condition, state)
        {expr, state} = normalize(expr, state)

        pair =
          quote do
            unquote(@cond_var_ast) = unquote(condition)
            {unquote(@cond_var_ast), fn -> unquote(expr) end}
          end

        {{meta, pair}, state}
      end)

    ast =
      quote do
        Nx.Defn.Expr.defn_cond(
          unquote(state.file),
          unquote(clauses)
        )
      end

    {ast, state}
  end

  defp normalize({name, meta, args}, state) when is_atom(name) and is_list(args) do
    arity = length(args)
    pair = {name, arity}

    cond do
      pair in state.defns or pair in state.transforms ->
        {args, state} = normalize_list(args, state)
        {{name, meta, args}, state}

      Module.defines?(state.module, {name, arity}) ->
        compile_error!(
          meta,
          state,
          "cannot use function #{name}/#{arity} inside defn because it was not defined with defn"
        )

      true ->
        compile_error!(
          meta,
          state,
          "undefined function #{name}/#{arity} (there is no such import)"
        )
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

  # TODO: Remove me once transform/2 is removed.
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

  defp normalize({{:., _, [:erlang, :error]} = dot, meta, args}, state) do
    {args, state} = normalize_list(args, state)
    {{dot, meta, args}, state}
  end

  defp normalize({{:., _, [_, :exception]} = dot, meta, [arg]}, state) do
    {arg, state} = normalize(arg, state)
    {{dot, meta, [arg]}, state}
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

  defp normalize({{:., dot_meta, [remote, name]}, meta, args}, state)
       # TODO: Remove args == [] once we require Elixir version where args are nil
       when is_atom(name) and (args == nil or args == []) do
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

  ## Params manipulation

  for i <- 0..128 do
    args = Macro.generate_arguments(i, __MODULE__)

    def fun(unquote(i), callback) do
      fn unquote_splicing(args) -> callback.(unquote(args)) end
    end
  end

  @doc false
  def to_lazy_params(fun, args) do
    {params, cache, {funs, _}} =
      Enum.reduce(args, {[], [], {[], 0}}, fn
        arg, {params, cache, acc}
        when is_list(arg)
        when is_function(arg)
        when is_tuple(arg) and is_function(elem(arg, 0)) ->
          {params, [arg | cache], acc}

        container, {params, cache, acc} ->
          {param, acc} =
            Nx.LazyContainer.traverse(container, acc, fn template, fun, {acc, i} ->
              {Nx.Defn.Expr.parameter(template, :root, i), {[fun | acc], i + 1}}
            end)

          {[param | params], [nil | cache], acc}
      end)

    if Enum.all?(cache, &is_nil/1) do
      {fun, Enum.reverse(params), Enum.reverse(funs)}
    else
      cache = Enum.reverse(cache)
      fun = fun(length(params), &apply(fun, merge_cache(cache, &1)))
      {fun, Enum.reverse(params), Enum.reverse(funs)}
    end
  end

  defp merge_cache([nil | cache], [head | tail]), do: [head | merge_cache(cache, tail)]
  defp merge_cache([head | tail], params), do: [head | merge_cache(tail, params)]
  defp merge_cache([], []), do: []

  @doc false
  def to_lazy_template(args) do
    {template_args, funs} =
      Enum.map_reduce(args, [], fn container, acc ->
        Nx.LazyContainer.traverse(container, acc, fn template, fun, acc ->
          {template, [fun | acc]}
        end)
      end)

    {template_args, Enum.reverse(funs)}
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
