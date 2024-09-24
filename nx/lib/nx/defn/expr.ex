defmodule Nx.Defn.Expr do
  @doc """
  The expression used by `Nx.Defn.Compiler`.

  `Nx.Defn.Compiler` changes `Nx` default backend from `Nx.BinaryBackend`
  to `Nx.Defn.Expr`. It is a struct with the following fields:

    * `:id` - a unique identifier
    * `:op` - the operation name
    * `:args` - the operation arguments
    * `:context` - the context of the expression.
      The default context is `:root`.

  Convenience functions for traversing expressions and composite types
  can be found in `Nx.Defn.Composite` and `Nx.Defn.Tree`.

  ## Syntax nodes

  Most nodes are created directly via the `Nx` module and
  therefore map directly to `Nx.Tensor` callbacks. However
  the following syntax nodes exist:

    * `parameter(integer)`

    * `constant(number)`

    * `tensor(tensor)`

    * `metadata(expr, metadata)`

    * `elem(tuple, pos)` - created automatically from
      expression that return tuples. Note it may return
      tuples too, which means we have nested tuples

    * `fun(parameters, t, mfa)` - the `mfa` is used only for
      introspection purposes

    * `cond(clauses, otherwise)`

    * `while(initial, condition, body)`

    * `attach_token(token(%Nx.Defn.Token{}), expr)`

  `defn` compilers must handle said nodes accordingly.
  """

  alias Nx.Defn.{Composite, Expr}
  alias Nx.Tensor, as: T

  import Nx.Shared

  @enforce_keys [:id, :op, :args, :context]
  defstruct [:id, :op, :args, :context]

  ## Public API

  @doc """
  Builds an tensor expression from the given tensor.
  """
  def tensor(tensor), do: to_expr(tensor)

  @doc """
  Creates a tensor expression parameter at `pos` based on the given tensor expression.
  """
  def parameter(%T{data: %Expr{context: context}} = tensor, pos) do
    parameter(tensor, context, pos)
  end

  @doc """
  Creates a tensor expression parameter at `pos` based on the given `tensor` and `context`.
  """
  def parameter(tensor, context, pos) when is_integer(pos) and pos >= 0 do
    expr(tensor, context, :parameter, [pos])
  end

  @doc """
  Creates a tensor expression parameter at `pos` with the given `context`, `type`,
  `shape`, and `pos`.
  """
  def parameter(context, type, shape, pos) do
    names = List.duplicate(nil, tuple_size(shape))
    expr(%T{type: type, shape: shape, names: names}, context, :parameter, [pos])
  end

  @doc """
  Creates a tensor expression metadata node wrapping the
  given tensor expression.

  The metadata is map. If the `inspect` key is present,
  it will be used to annotate the metadata when inspected.
  Otherwise the metadata node does not appear during
  inspection.
  """
  def metadata(expr, metadata) when is_map(metadata) do
    case to_container_expr(expr) do
      %{data: %{context: context}} = res ->
        expr(res, context, :metadata, [Nx.devectorize(expr), metadata])

      t when is_tuple(t) ->
        context = elem(t, 0).data.context

        tuple(
          expr(tuple_out(tuple_size(t)), context, :metadata, [Nx.devectorize(expr), metadata]),
          Tuple.to_list(t)
        )
    end
  end

  @doc """
  Creates a tuple with elements in `list` that points to tuple
  expression `expr`.

  `list` must be a list of tensor expressions of the same size
  as the tuple expression.
  """
  def tuple(%T{type: {:tuple, size}, data: %{context: context}} = expr, list)
      when is_list(list) do
    tuple =
      list
      |> Enum.with_index(fn %T{} = tensor, i ->
        expr(tensor, context, :elem, [expr, i])
      end)
      |> List.to_tuple()

    ^size = tuple_size(tuple)
    tuple
  end

  @doc """
  Creates a `cond` tensor expression.
  """
  def cond([], last), do: last

  def cond(clauses, last) do
    {preds, exprs} = Enum.unzip(clauses)
    {preds, context} = to_exprs(preds)

    if Enum.all?(preds, &(&1.vectorized_axes == [])) do
      non_vectorized_cond(clauses, last, context)
    else
      # We reshape_vectors to ensure that all predicate vectorized axes are encountered
      # in the same order throughout the clauses. the zip_with below ensures that only
      # the vectorized predicates are pushed through with this new order
      reshaped_preds = Nx.reshape_vectors(preds, align_ranks: true)

      clauses =
        Enum.zip_with([preds, reshaped_preds, exprs], fn
          [%{vectorized_axes: []} = pred, _reshaped, expr] -> {pred, expr}
          [_, reshaped, expr] -> {reshaped, expr}
        end)

      # note: because we already checked above if any pred is vectorized, the right
      # side of the tuple will always have at least one element in the list
      {non_vectorized_pre, [{pred, expr} | other]} =
        Enum.split_while(clauses, fn {pred, _expr} -> pred.vectorized_axes == [] end)

      # `other` can also be an empty list. `cond([], last)` deals with this above
      devec_pred = Nx.devectorize(pred)
      then_selector = Nx.any(devec_pred)

      # note: any(not(x)) == not(all(x)); we can skip the outer not if we swap the branches
      else_selector = Nx.all(devec_pred)

      zero =
        Composite.traverse(last, fn leaf ->
          vectorized_axes = Nx.to_tensor(leaf).vectorized_axes

          Nx.broadcast(0, Nx.devectorize(leaf).shape) |> Nx.vectorize(vectorized_axes)
        end)

      # here we build the `other` clauses into a remaining cond that also contains `last`.
      # It will also be the default value for when the expr is not executed
      then_clause = non_vectorized_cond([{then_selector, expr}], zero, context)

      else_clause =
        Composite.flatten_list([
          non_vectorized_cond([{else_selector, zero}], cond(other, last), context)
        ])

      {clause, []} =
        Composite.traverse(then_clause, else_clause, fn
          l, [r | tl] ->
            {Nx.select(pred, l, r), tl}
        end)

      non_vectorized_cond(non_vectorized_pre, clause, context)
    end
  end

  defp non_vectorized_cond([], last, _context) do
    last
  end

  defp non_vectorized_cond(clauses, out_template = last, context) do
    {preds, exprs} = Enum.unzip(clauses)

    {broadcasted_clauses, vectorized_axes} =
      [last | exprs]
      |> Enum.map(&Composite.flatten_list([&1]))
      |> Enum.zip_with(&broadcast_clause/1)
      |> Enum.unzip()

    [last | exprs] =
      case broadcasted_clauses do
        # Handle the case where branches don't return anything
        [] -> Enum.map([last | exprs], fn _ -> {} end)
        clauses -> unzip_clauses(clauses)
      end

    clauses = Enum.zip(preds, exprs)

    out =
      if vectorized_axes == [] do
        out_template
      else
        {result, []} =
          Composite.traverse(out_template, vectorized_axes, fn
            %T{} = expr, [axes | tail] ->
              {%{expr | vectorized_axes: axes}, tail}

            expr, [[] | tail] ->
              {expr, tail}
          end)

        result
      end

    flatten_to_composite(
      out,
      context,
      exprs,
      &expr(&1, context, :cond, [clauses, last])
    )
  end

  defp broadcast_clause([type = last | expr_types = exprs]) do
    [%{vectorized_axes: vectorized_axes} = last | exprs] =
      Nx.reshape_vectors([last | exprs], align_ranks: true)

    [%{shape: shape, names: names} = last | exprs] = Enum.map([last | exprs], &Nx.devectorize/1)

    {exprs, {type, shape, names}} =
      Enum.map_reduce(Enum.zip(exprs, expr_types), {type, shape, names}, fn {expr, expr_type},
                                                                            {type, shape, names} ->
        type = binary_type(type, expr_type)
        {shape, names} = Nx.Shape.binary_broadcast(shape, names, expr.shape, expr.names)
        {expr, {type, shape, names}}
      end)

    result =
      for expr <- [last | exprs] do
        expr
        |> Nx.as_type(type)
        |> Nx.broadcast(shape, names: names)
      end

    {result, vectorized_axes}
  end

  defp unzip_clauses([exprs | _] = clauses),
    do: unzip_clauses(clauses, List.duplicate([], length(exprs)))

  defp unzip_clauses([exprs | tail], acc),
    do: unzip_clauses(tail, unzip_each(exprs, acc))

  defp unzip_clauses([], acc) do
    Enum.map(acc, fn
      [entry] -> entry
      list -> List.to_tuple(Enum.reverse(list))
    end)
  end

  defp unzip_each([head | tail], [acc_head | acc_tail]),
    do: [[head | acc_head] | unzip_each(tail, acc_tail)]

  defp unzip_each([], []),
    do: []

  @doc """
  Creates a `while` tensor expression.
  """
  def while(initial, context, arg, condition, body) do
    [flatten_initial, flatten_arg, flatten_body] = clauses = flatten_clauses([initial, arg, body])

    if condition.vectorized_axes != [] do
      raise ArgumentError,
            "condition for while cannot be vectorized, got: #{inspect(condition)}"
    end

    args = [flatten_initial, flatten_arg, condition, flatten_body]
    flatten_to_composite(initial, context, clauses, &expr(&1, context, :while, args))
  end

  defp flatten_clauses(clauses) do
    Enum.map(clauses, fn expr ->
      case Composite.flatten_list([expr]) do
        [single] -> Nx.devectorize(single)
        list -> list |> Enum.map(&Nx.devectorize/1) |> List.to_tuple()
      end
    end)
  end

  defp flatten_to_composite(out, context, [head | _], fun) when is_tuple(head) do
    size = tuple_size(head)
    expr = fun.(tuple_out(size))

    vectorized_axes =
      out
      |> Composite.reduce([], fn
        %T{vectorized_axes: [_ | _] = axes}, acc ->
          [Keyword.keys(axes) | acc]

        _t, acc ->
          [nil | acc]
      end)
      |> Enum.reverse()

    {out, {[], ^size, []}} =
      Composite.traverse(out, {Tuple.to_list(head), 0, vectorized_axes}, fn
        _, {[head | tail], i, [axes | axes_tail]} ->
          head =
            if axes do
              Nx.vectorize(head, axes)
            else
              head
            end

          {expr(head, context, :elem, [expr, i]), {tail, i + 1, axes_tail}}
      end)

    out
  end

  defp flatten_to_composite(out, _context, [head | _], fun) do
    vectorized_axes =
      out
      |> Composite.reduce([], fn
        %T{vectorized_axes: [_ | _] = axes}, acc ->
          [Keyword.keys(axes) | acc]

        _, acc ->
          [nil | acc]
      end)
      |> Enum.reverse()

    {out, {[], []}} =
      Composite.traverse(out, {[fun.(head)], vectorized_axes}, fn
        _, {[head | tail], [axes | axes_tail]} ->
          head =
            if axes do
              Nx.vectorize(head, axes)
            else
              head
            end

          {head, {tail, axes_tail}}
      end)

    out
  end

  @impl true
  def optional(name, in_args, fun) do
    {args, opts} = Enum.split_while(in_args, &(not is_list(&1)))
    params = Enum.with_index(args, &parameter/2)

    case apply(fun, params ++ opts) do
      %{data: %{context: context}} = res ->
        expr(res, context, :optional, [expr(res, context, name, in_args), res, fun])

      t when is_tuple(t) ->
        context = elem(t, 0).data.context
        out = expr(tuple_out(tuple_size(t)), context, name, in_args)
        tuple(expr(out, context, :optional, [out, t, fun]), Tuple.to_list(t))
    end
  end

  ## Nx.Defn AST callbacks

  @doc false
  def id(), do: make_ref()

  @doc false
  def add_hook(token, expr, name, function) do
    expr = to_container_expr(expr)
    token = Nx.Defn.Token.add_hook(token, expr, name, function)
    {token, expr}
  end

  @doc false
  def attach_token(%T{data: %{op: :token}} = token, expr) do
    Composite.traverse(expr, fn tensor ->
      expr = to_expr(tensor)
      expr(expr, expr.data.context, :attach_token, [token, expr])
    end)
  end

  def attach_token(%Nx.Defn.Token{} = token, expr) do
    # We first create an expression to store the token
    # so we have a shared ID to avoid multiple traversals.
    # The size of the tuple is not used, but the amount of
    # hooks is a good indicator.
    size = length(token.hooks)
    token = expr(%T{shape: {}, type: {:tuple, size}, names: []}, nil, :token, [token])
    attach_token(token, expr)
  end

  @doc false
  def defn_cond(file, [{meta, _} | _] = clauses) do
    clauses =
      for {meta, {pred, expr}} <- clauses,
          pred = to_pred(pred, meta[:line], file, :cond),
          # Eliminate all clauses that will never match
          not match?(%T{data: %Expr{op: :constant, args: [number]}} when number == 0, pred) do
        {meta, pred, expr}
      end

    case clauses do
      # At least one clause is expected
      [] ->
        raise CompileError,
          line: meta[:line],
          file: file,
          description: "cond/if expects at least one branch to always evaluate to true"

      # We found a clause that always matches, return it always
      [{_meta, %T{data: %Expr{op: :constant, args: [number]}}, expr} | _] when number != 0 ->
        expr.()

      # Otherwise, keep it as a cond and validate the last clause always returns true
      [{_, first_pred, first} | rest] ->
        first = first.()

        {[{last_pred, last} | reverse], _} =
          Enum.reduce(rest, {[{first_pred, first}], 1}, fn {meta, pred, expr},
                                                           {acc, branch_idx} ->
            expr = expr.()

            if not Nx.Defn.Composite.compatible?(first, expr, fn _, _ -> true end) do
              raise CompileError,
                line: meta[:line],
                file: file,
                description: """
                cond/if expects all branches to return compatible tensor types.

                #{Nx.Defn.TemplateDiff.build_and_inspect(first, expr, "First Branch (expected)", "Branch #{branch_idx}", fn _, _ -> true end)}
                """
            end

            {[{pred, expr} | acc], branch_idx + 1}
          end)

        case last_pred do
          %T{data: %Expr{op: :constant, args: [number]}} when number != 0 ->
            cond(Enum.reverse(reverse), last)

          _ ->
            raise CompileError,
              line: meta[:line],
              file: file,
              description: "cond/if expects at least one branch to always evaluate to true"
        end
    end
  end

  @doc false
  def defn_while(file, line, initial, generator, condition_body, opts) do
    initial = to_container_expr(initial)

    case generator do
      :none ->
        if opts != [] do
          raise CompileError,
            line: line,
            file: file,
            description:
              "options are not supported on while with conditions, only with generators, got: #{inspect(opts)}"
        end

        {arg, context} = to_param_expr(initial, :while)
        condition = condition_body.(:condition, arg) |> to_pred(line, file, :while)
        while_vectorized(file, line, initial, context, arg, condition, condition_body)

      {:while, %T{} = generator} ->
        range =
          case Nx.shape(generator) do
            {} ->
              message = "cannot have a scalar tensor as generator"
              raise CompileError, line: line, file: file, description: message

            _ ->
              0..(Nx.axis_size(generator, 0) - 1)//1
          end

        condition_body = fn which, {{index, generator_expr}, acc} ->
          condition_body.(which, {generator_expr[index], acc})
        end

        while_range(range, file, line, initial, generator, condition_body, opts)

      {:while, _.._//_ = range} ->
        condition_body = fn which, {{index, {}}, acc} ->
          condition_body.(which, {index, acc})
        end

        while_range(range, file, line, initial, {}, condition_body, opts)

      {:while, other} ->
        raise CompileError,
          line: line,
          file: file,
          description: "generators in while expect a range or a tensor, got: #{inspect(other)}"
    end
  end

  defp while_vectorized(file, line, initial, context, arg, condition, condition_body) do
    case condition.vectorized_axes do
      [] ->
        body = condition_body.(:body, arg) |> to_container_expr()
        compatible_while!(file, line, initial, body)
        while(initial, context, arg, condition, body)

      [_ | _] ->
        # 1. broadcast_vectors each `initial` arg with `pred`
        # 2. Flatten the common vectorized axes
        # 3. add a parameter "i" for iterating on an outer while
        # 4. take the i-th position and pass to the inner while
        # 5. put_slice the results onto the accumulators
        # 6. return the tuple without the "i" parameter

        {initial, max_vec_size} = vectorized_while__vectorize_initial(initial, condition)

        outer_initial = {tensor(0), initial}
        {arg, outer_context} = to_param_expr(outer_initial, :while)
        {index_param, outer_arg} = arg
        outer_condition = Nx.less(index_param, tensor(max_vec_size))
        inner_initial = vectorized_while__take_index_param(outer_arg, index_param)
        {inner_arg, inner_context} = to_param_expr(inner_initial, :while)
        inner_condition = condition_body.(:condition, inner_arg) |> to_pred(line, file, :while)
        inner_body = condition_body.(:body, inner_arg) |> to_container_expr()

        compatible_while!(file, line, inner_initial, inner_body)

        inner_while = while(inner_initial, inner_context, inner_arg, inner_condition, inner_body)

        vectorized_while__build_outer_while(
          condition.vectorized_axes,
          outer_initial,
          outer_context,
          arg,
          outer_condition,
          inner_while,
          outer_arg,
          index_param,
          Nx.add(index_param, 1)
        )
    end
  end

  defp while_range(
         range,
         file,
         line,
         initial,
         %T{vectorized_axes: [{first_axis, _} | _] = vectorized_axes} = generator,
         condition_body,
         opts
       ) do
    {initial, max_vec_size} = vectorized_while__vectorize_initial(initial, generator)

    flat_generator = generator |> Nx.revectorize([{first_axis, :auto}]) |> Nx.devectorize()
    outer_initial = {{tensor(0), flat_generator}, initial}
    {arg, outer_context} = to_param_expr(outer_initial, :while)
    {{index_param, inner_gen_param}, outer_arg} = arg
    outer_condition = Nx.less(index_param, tensor(max_vec_size))
    inner_initial = vectorized_while__take_index_param(outer_arg, index_param)
    inner_cond = Nx.take(inner_gen_param, index_param)
    inner_while = while_range(range, file, line, inner_initial, inner_cond, condition_body, opts)

    result =
      vectorized_while__build_outer_while(
        vectorized_axes,
        outer_initial,
        outer_context,
        arg,
        outer_condition,
        inner_while,
        outer_arg,
        index_param,
        {Nx.add(index_param, 1), inner_gen_param}
      )

    Composite.traverse(result, fn leaf ->
      %{vectorized_axes: [{^first_axis, _} | other_axes]} = leaf
      Nx.revectorize(leaf, vectorized_axes ++ other_axes)
    end)
  end

  defp while_range(range, file, line, initial, generator, condition_body, opts) do
    opts = Keyword.validate!(opts, unroll: false)
    size = Range.size(range)

    {internal, external} =
      case opts[:unroll] do
        true ->
          {nil, range}

        false ->
          {{range, 0..0//1}, 0..-1//1}

        unroll when is_integer(unroll) and unroll >= size ->
          {nil, range}

        unroll when is_integer(unroll) and unroll > 0 ->
          {internal, external} = Range.split(range, size - rem(size, unroll))
          {{internal, 0..(unroll - 1)//1}, external}

        unroll ->
          message = ":unroll must be a boolean, got: #{inspect(unroll)}"
          raise CompileError, line: line, file: file, description: message
      end

    result =
      case internal do
        {first..last//step, internal_unroll} ->
          gen_initial = {{tensor(first), generator}, initial}
          {{{index_param, generator_param}, arg}, context} = to_param_expr(gen_initial, :while)

          condition =
            if step > 0 do
              Nx.less_equal(index_param, tensor(last))
            else
              Nx.greater_equal(index_param, tensor(last))
            end

          body =
            Enum.reduce(internal_unroll, arg, fn index, acc ->
              next = Nx.add(index_param, step * index)
              body = condition_body.(:body, {{next, generator_param}, acc}) |> to_container_expr()
              index == 0 and compatible_while!(file, line, initial, body)
              body
            end)

          next = Range.size(internal_unroll) * step
          gen_arg = {{index_param, generator_param}, arg}
          gen_body = {{Nx.add(index_param, next), generator_param}, body}
          {_, result} = while(gen_initial, context, gen_arg, condition, gen_body)
          result

        nil ->
          initial
      end

    Enum.reduce(external, result, fn index, acc ->
      body = condition_body.(:body, {{index, generator}, acc}) |> to_container_expr()
      index == external.first and compatible_while!(file, line, initial, body)
      body
    end)
  end

  defp vectorized_while__vectorize_initial(initial, target) do
    {first_axis, _} = hd(target.vectorized_axes)
    num_vectorized_axes = length(target.vectorized_axes)

    Composite.traverse(initial, nil, fn leaf, _ ->
      # broadcast and pull common axes to the front
      [_, leaf] = Nx.broadcast_vectors([target, leaf], align_ranks: true)

      %{vectorized_axes: leaf_axes} = leaf

      # remove common axes so that we can flatten them into a fixed axis.
      # We use the `first_axis` to ensure that there's no name collision
      # in an easy way.
      vectorized_axes = [{first_axis, :auto} | Enum.drop(leaf_axes, num_vectorized_axes)]

      %{vectorized_axes: [{_, s} | _]} = leaf = Nx.revectorize(leaf, vectorized_axes)
      {leaf, s}
    end)
  end

  defp vectorized_while__take_index_param(outer_arg, index_param) do
    Composite.traverse(outer_arg, fn %{vectorized_axes: [_ | ax]} = leaf ->
      leaf
      |> Nx.devectorize()
      |> Nx.take(index_param)
      |> Nx.vectorize(ax)
    end)
  end

  defp vectorized_while__build_outer_while(
         [{first_axis, _} | _] = vectorized_axes,
         outer_initial,
         outer_context,
         arg,
         outer_condition,
         inner_while,
         outer_arg,
         index_param,
         index_param_update_rule
       ) do
    {outer_body, _} =
      Composite.traverse(
        inner_while,
        Composite.flatten_list([outer_arg]),
        fn node, [target | targets] ->
          target = Nx.devectorize(target)
          starts = [index_param | List.duplicate(0, tuple_size(target.shape) - 1)]

          updated_target =
            target
            |> Nx.put_slice(starts, Nx.devectorize(node) |> Nx.new_axis(0))
            |> Nx.vectorize(target.vectorized_axes)

          {updated_target, targets}
        end
      )

    {_aux_index, result} =
      while(
        outer_initial,
        outer_context,
        arg,
        outer_condition,
        {index_param_update_rule, outer_body}
      )

    Composite.traverse(result, fn leaf ->
      %{vectorized_axes: [{^first_axis, _} | other_axes]} = leaf
      Nx.revectorize(leaf, vectorized_axes ++ other_axes)
    end)
  end

  defp compatible_while!(file, line, initial, body) do
    if not Nx.compatible?(initial, body) do
      raise CompileError,
        line: line,
        file: file,
        description: """
        the do-block in while must return tensors with the same shape, type, and names as the initial arguments.

        #{Nx.Defn.TemplateDiff.build_and_inspect(body, initial, "Body (do-block)", "Initial")}
        """
    end
  end

  ## Nx.Backend Callbacks

  @behaviour Nx.Backend

  @impl true
  def init(opts) do
    if opts != [] do
      raise ArgumentError, "Nx.BinaryBackend accepts no options"
    end

    opts
  end

  @impl true
  def from_binary(binary, type, _options) do
    to_expr(Nx.BinaryBackend.from_binary(binary, type, []))
  end

  @impl true
  def constant(out, number, _options) do
    constant(out, number)
  end

  @impl true
  def eye(out, _backend_options) do
    expr(out, nil, :eye, [])
  end

  @impl true
  def iota(out, axis, _backend_options) do
    expr(out, nil, :iota, [axis])
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
      [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
      [:is_nan, :is_infinity] ++
      [:conjugate, :population_count, :count_leading_zeros, :floor, :ceil, :round] ++
      [:erf, :erfc, :erf_inv, :acos, :asin, :atan, :bitcast, :real, :imag]

  for op <- unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      tensor = to_expr(tensor)
      unary_expr(out, tensor.data.context, unquote(op), tensor)
    end
  end

  @impl true
  def add(out, t1, t2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    c1 = maybe_constant(t1)
    c2 = maybe_constant(t2)

    cond do
      c1 == 0 ->
        ensure_compatible(t2, out)

      c2 == 0 ->
        ensure_compatible(t1, out)

      c2 ->
        commute(out, context, :add, &Complex.add/2, c2, t2, t1)

      true ->
        case t2 do
          %T{
            data: %Expr{
              op: :subtract,
              args: [%T{data: %Expr{op: :constant, args: [constant]}}, t2]
            }
          }
          when constant == 0 ->
            binary_expr(out, context, :subtract, t1, t2)

          %T{} ->
            commute(out, context, :add, &Complex.add/2, c1, t1, t2)
        end
    end
  end

  @impl true
  def subtract(out, t1, t2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    c1 = maybe_constant(t1)
    c2 = maybe_constant(t2)

    cond do
      c2 == 0 -> ensure_compatible(t1, out)
      c1 && c2 -> constant(out, Complex.subtract(c1, c2))
      true -> binary_expr(out, context, :subtract, t1, t2)
    end
  end

  @impl true
  def multiply(out, t1, t2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    c1 = maybe_constant(t1)
    c2 = maybe_constant(t2)

    cond do
      c1 == 1 ->
        ensure_compatible(t2, out)

      c2 == 1 ->
        ensure_compatible(t1, out)

      c2 ->
        commute(out, context, :multiply, &Complex.multiply/2, c2, t2, t1)

      true ->
        case t2 do
          %T{
            data: %Expr{op: :divide, args: [%T{data: %Expr{op: :constant, args: [constant]}}, t2]}
          }
          when constant == 1 ->
            binary_expr(out, context, :divide, t1, t2)

          %T{} ->
            commute(out, context, :multiply, &Complex.multiply/2, c1, t1, t2)
        end
    end
  end

  @impl true
  def divide(out, t1, t2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    c2 = maybe_constant(t2)

    cond do
      c2 == 1 -> ensure_compatible(t1, out)
      true -> binary_expr(out, context, :divide, t1, t2)
    end
  end

  @impl true
  def pow(out, t1, t2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    c2 = maybe_constant(t2)

    cond do
      c2 == 1 -> ensure_compatible(t1, out)
      true -> binary_expr(out, context, :pow, t1, t2)
    end
  end

  binary_ops =
    [:remainder, :atan2, :max, :min, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :less_equal, :greater_equal] ++
      [:logical_and, :logical_or, :logical_xor]

  for op <- binary_ops do
    @impl true
    def unquote(op)(out, t1, t2) do
      {[t1, t2], context} = to_exprs([t1, t2])
      binary_expr(out, context, unquote(op), t1, t2)
    end
  end

  aggregate_ops = [:all, :any, :argmax, :argmin, :sum, :product, :reduce_min, :reduce_max]

  for op <- aggregate_ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor, opts])
    end
  end

  window_aggregate_ops = [:window_sum, :window_product, :window_max, :window_min]

  for op <- window_aggregate_ops do
    @impl true
    def unquote(op)(out, tensor, window_dimensions, opts) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor, window_dimensions, opts])
    end
  end

  @impl true
  def reduce(%{type: type} = out, tensor, acc, opts, fun) do
    context = new_context(:reduce)
    args = [parameter(context, type, {}, 0), parameter(context, type, {}, 1)]
    {[tensor, acc], context} = to_exprs([tensor, acc])
    fun = apply_fun(context, fun, args, type)

    if fun.shape != {} do
      raise "reduce function must return a scalar tensor, got: #{inspect(fun.shape)}"
    end

    expr(out, context, :reduce, [tensor, acc, opts, fun])
  end

  @impl true
  def window_reduce(
        %{type: type} = out,
        tensor,
        acc,
        window_dims,
        opts,
        fun
      ) do
    context = new_context(:window_reduce)
    args = [parameter(context, type, {}, 0), parameter(context, type, {}, 1)]
    {[tensor, acc], context} = to_exprs([tensor, acc])
    fun = apply_fun(context, fun, args, type)

    if fun.shape != {} do
      raise "window_reduce function must return a scalar tensor, got: #{inspect(fun.shape)}"
    end

    expr(out, context, :window_reduce, [tensor, acc, window_dims, opts, fun])
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, window_dims, opts) do
    {[tensor, source, init_value], context} = to_exprs([tensor, source, init_value])

    args = [tensor, source, init_value, window_dims, opts]
    expr(out, context, :window_scatter_max, args)
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, window_dims, opts) do
    {[tensor, source, init_value], context} = to_exprs([tensor, source, init_value])

    args = [tensor, source, init_value, window_dims, opts]
    expr(out, context, :window_scatter_min, args)
  end

  @impl true
  def indexed_add(out, target, indices, updates, opts) do
    {[target, indices, updates], context} = to_exprs([target, indices, updates])

    expr(out, context, :indexed_add, [target, indices, updates, opts])
  end

  @impl true
  def indexed_put(out, target, indices, updates, opts) do
    {[target, indices, updates], context} = to_exprs([target, indices, updates])

    expr(out, context, :indexed_put, [target, indices, updates, opts])
  end

  @impl true
  def reshape(out, tensor) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :reshape, [tensor])
  end

  @impl true
  def squeeze(out, tensor, axes) do
    tensor = to_expr(tensor)

    # If we are in a sequence of squeezes, we collapse them.
    # This helps us fuse the access syntax.
    with %T{data: %Expr{op: :squeeze, args: [tensor, inner_axes]}} <- tensor do
      axes = merge_squeeze(Enum.sort(inner_axes), Enum.sort(axes), 0)
      expr(out, tensor.data.context, :squeeze, [tensor, axes])
    else
      _ -> expr(out, tensor.data.context, :squeeze, [tensor, axes])
    end
  end

  defp merge_squeeze([inner_axis | inner_axes], [axis | axes], extra)
       when inner_axis <= axis + extra,
       do: [inner_axis | merge_squeeze(inner_axes, [axis | axes], extra + 1)]

  defp merge_squeeze(inner_axes, [axis | axes], extra),
    do: [axis + extra | merge_squeeze(inner_axes, axes, extra)]

  defp merge_squeeze([], [], _extra),
    do: []

  @impl true
  def transpose(out, tensor, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :transpose, [tensor, axes])
  end

  @impl true
  def as_type(out, tensor) do
    tensor = to_expr(tensor)
    unary_expr(out, tensor.data.context, :as_type, tensor)
  end

  @impl true
  def broadcast(out, tensor, shape, axes) do
    tensor = to_expr(tensor)

    with %T{data: %Expr{op: :broadcast, args: [inner_tensor, inner_shape, inner_axes]}} <- tensor,
         true <-
           (contiguous?(inner_axes, 0) and contiguous?(axes, 0)) or
             (contiguous_last?(inner_axes, inner_shape, inner_tensor) and
                contiguous_last?(axes, shape, tensor)) do
      expr(out, tensor.data.context, :broadcast, [inner_tensor, shape, inner_axes])
    else
      _ ->
        if c = maybe_constant(tensor) do
          constant(out, c)
        else
          expr(out, tensor.data.context, :broadcast, [tensor, shape, axes])
        end
    end
  end

  defp contiguous_last?(axes, out_shape, in_shape),
    do: contiguous?(axes, Nx.rank(out_shape) - Nx.rank(in_shape))

  defp contiguous?([], _), do: true
  defp contiguous?([i | rest], i), do: contiguous?(rest, i + 1)
  defp contiguous?(_, _), do: false

  @impl true
  def dot(out, t1, c1, b1, t2, c2, b2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    expr(out, context, :dot, [t1, c1, b1, t2, c2, b2])
  end

  @impl true
  def conv(out, inp, kernel, opts) do
    {[inp, kernel], context} = to_exprs([inp, kernel])
    expr(out, context, :conv, [inp, kernel, opts])
  end

  @impl true
  def pad(out, expr, value, config) do
    {[expr, value], context} = to_exprs([expr, value])
    expr(out, context, :pad, [expr, value, config])
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    {[pred, on_true, on_false], context} = to_exprs([pred, on_true, on_false])
    expr(out, context, :select, [pred, on_true, on_false])
  end

  @impl true
  def clip(out, operand, min, max) do
    {[operand, min, max], context} = to_exprs([operand, min, max])
    expr(out, context, :clip, [operand, min, max])
  end

  @impl true
  def slice(out, tensor, start, lengths, strides) do
    all_static? = Enum.all?(start, &is_integer/1)

    {[tensor | start], context} =
      if all_static? do
        tensor = to_expr(tensor)
        {[tensor | start], tensor.data.context}
      else
        to_exprs([tensor | start])
      end

    # If we are in a sequence of slices, it is the access syntax,
    # so we compact them into a single slice.
    with true <- ones_stride?(strides),
         {slice, axes} <- maybe_squeeze(tensor),
         %T{data: %Expr{op: :slice, args: [tensor, inner_start, inner_lengths, strides]}} <-
           slice,
         true <- ones_stride?(strides) do
      {start, lengths} =
        0
        |> merge_slice(axes, inner_start, start, inner_lengths, lengths)
        |> Enum.unzip()

      tensor
      |> Nx.slice(start, lengths)
      |> Nx.squeeze(axes: axes)
    else
      _ ->
        expr(out, context, :slice, [tensor, start, lengths, strides])
    end
  end

  defp ones_stride?(strides), do: Enum.all?(strides, &(&1 == 1))

  defp maybe_squeeze(%T{data: %Expr{op: :squeeze, args: [slice, axes]}}), do: {slice, axes}
  defp maybe_squeeze(slice), do: {slice, []}

  defp merge_slice(_axis, _axes, [], [], [], []), do: []

  defp merge_slice(axis, axes, [is | inner_start], start, [il | inner_lengths], lengths) do
    # This is one of the erased axes, so we need to get coordinates from inner
    if axis in axes do
      [{is, il} | merge_slice(axis + 1, axes, inner_start, start, inner_lengths, lengths)]
    else
      [s | start] = start
      [l | lengths] = lengths

      [
        {Nx.Defn.Kernel.+(is, s), l}
        | merge_slice(axis + 1, axes, inner_start, start, inner_lengths, lengths)
      ]
    end
  end

  @impl true
  def put_slice(out, tensor, start, slice) do
    {[tensor, slice | start], context} = to_exprs([tensor, slice | start])

    expr(out, context, :put_slice, [tensor, start, slice])
  end

  @impl true
  def gather(out, tensor, indices, opts) do
    {[tensor, indices], context} = to_exprs([tensor, indices])
    expr(out, context, :gather, [tensor, indices, opts])
  end

  @impl true
  def reverse(out, tensor, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :reverse, [tensor, axes])
  end

  @impl true
  def concatenate(out, tensors, axis) do
    {tensors, context} = to_exprs(tensors)
    expr(out, context, :concatenate, [tensors, axis])
  end

  @impl true
  def stack(out, tensors, axis) do
    {tensors, context} = to_exprs(tensors)
    expr(out, context, :stack, [tensors, axis])
  end

  @impl true
  def triangular_solve(out, a, b, opts) do
    {[a, b], context} = to_exprs([a, b])
    expr(out, context, :triangular_solve, [a, b, opts])
  end

  @impl true
  def lu({p, l, u}, tensor, opts) do
    tensor = to_expr(tensor)
    context = tensor.data.context
    out = %T{names: [], shape: {}, type: {:tuple, 3}}
    tuple(expr(out, context, :lu, [{p, l, u}, tensor, opts]), [p, l, u])
  end

  @impl true
  def sort(out, tensor, opts) do
    %{data: %{context: context}} = tensor = to_expr(tensor)
    expr(out, context, :sort, [tensor, opts])
  end

  @impl true
  def argsort(out, tensor, opts) do
    %{data: %{context: context}} = tensor = to_expr(tensor)
    expr(out, context, :argsort, [tensor, opts])
  end

  @impl true
  def fft(out, tensor, opts) do
    %{data: %{context: context}} = tensor = to_expr(tensor)
    expr(out, context, :fft, [tensor, opts])
  end

  @impl true
  def ifft(out, tensor, opts) do
    %{data: %{context: context}} = tensor = to_expr(tensor)
    expr(out, context, :ifft, [tensor, opts])
  end

  ## Undefined

  @impl true
  def backend_transfer(out, __MODULE__, _), do: out

  ops = [
    backend_copy: 3,
    backend_deallocate: 1,
    backend_transfer: 3,
    to_binary: 2,
    to_batched: 3,
    from_pointer: 5,
    to_pointer: 2
  ]

  for {op, arity} <- ops do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(op)(unquote_splicing(args)) do
      raise ArgumentError, """
      cannot invoke #{unquote(op)}/#{unquote(arity)} on Nx.Defn.Expr.

      This typically means you are invoking an unsupported Nx function
      inside `defn` or inside JIT compiled code
      """
    end
  end

  ## Helpers

  @compile {:inline, new_context: 1}
  defp new_context(atom) when is_atom(atom) do
    {atom, make_ref()}
  end

  defp expr(tensor, context, op, args) do
    %{tensor | data: %Expr{id: id(), op: op, args: args, context: context}}
  end

  defp to_expr(%T{data: %Expr{}} = t),
    do: t

  defp to_expr(%T{data: %Nx.BinaryBackend{}, shape: shape} = t) do
    case shape do
      {} -> constant(t, Nx.to_number(t))
      _ -> expr(t, nil, :tensor, [t])
    end
  end

  defp to_expr(%T{data: %backend{}} = t) do
    raise ArgumentError,
          "cannot convert tensor allocated on #{inspect(backend)} to an expression. " <>
            "This may mean you are passing a tensor to defn/jit as an optional argument " <>
            "or as closure in an anonymous function. For efficiency, it is preferred " <>
            "to always pass tensors as required arguments instead. Alternatively, you " <>
            "could call Nx.backend_copy/1 on the tensor, however this will copy its " <>
            "value and inline it inside the defn expression. Got: #{inspect(t)}"
  end

  defp to_expr(number) when is_number(number),
    do: constant(%T{shape: {}, names: [], type: Nx.Type.infer(number)}, number)

  defp to_expr(other) do
    raise ArgumentError,
          "unable to build tensor expression, expected a tensor or a number, " <>
            "got: #{inspect(other)}"
  end

  defp to_exprs(list) do
    Enum.map_reduce(list, nil, fn tensor, acc ->
      expr = to_expr(tensor)
      {expr, merge_context!(expr, acc)}
    end)
  end

  defp to_container_expr(container_or_tensor) do
    Composite.traverse(container_or_tensor, &to_expr/1)
  end

  defp to_param_expr(container_or_tensor, context_label) do
    context = new_context(context_label)

    {arg, {_, context}} =
      Composite.traverse(container_or_tensor, {0, nil}, fn expr, {counter, acc} ->
        {parameter(expr, context, counter), {counter + 1, merge_context!(expr, acc)}}
      end)

    {arg, context}
  end

  defp tuple_out(size) do
    %T{shape: {}, names: [], type: {:tuple, size}}
  end

  defp fun(context, args, body, {_, _, _} = mfa) do
    case to_container_expr(body) do
      %T{} = tensor ->
        expr(tensor, context, :fun, [args, tensor, mfa])

      tuple when is_tuple(tuple) ->
        expr(tuple_out(tuple_size(tuple)), context, :fun, [args, tuple, mfa])
    end
  end

  defp apply_fun(context, fun, args, type) when is_function(fun, length(args)) do
    {:module, mod} = Function.info(fun, :module)
    {:name, name} = Function.info(fun, :name)
    {:arity, arity} = Function.info(fun, :arity)

    # We modify the type after applying because the best form
    # to perform type conversions is always left to the compiler.
    %{fun(context, args, apply(fun, args), {mod, name, arity}) | type: type}
  end

  defp to_pred(pred, line, file, op) do
    pred =
      cond do
        is_boolean(pred) ->
          number = if pred == false, do: 0, else: 1
          %T{data: constant_expr({}, {:u, 8}, number), shape: {}, type: {:u, 8}, names: []}

        is_atom(pred) or is_binary(pred) or is_list(pred) ->
          raise CompileError,
            line: line,
            file: file,
            description:
              "#{Atom.to_string(op)} in defn expects the predicate to be true, false," <>
                " or a scalar tensor where 0 is false and everything else is true." <>
                " Unsupported value: #{inspect(pred)}"

        true ->
          to_expr(pred)
      end

    if not match?(%T{shape: {}}, pred) do
      raise CompileError,
        line: line,
        file: file,
        description:
          "condition must be a scalar tensor, got: #{inspect(pred)}," <>
            " consider using Nx.all/1 or Nx.any/1 to obtain a scalar" <>
            " predicate from tensor"
    end

    pred
  end

  defp merge_context!(%{data: %{context: context}}, acc) do
    if context != acc and context != nil and acc != nil do
      raise """
      cannot build defn because expressions come from different contexts: \
      #{inspect(context)} and #{inspect(acc)}.

      This typically happens on "while" and inside anonymous functions when you \
      try to access an external variable. All variables you intend to use inside \
      "while" or anonymous functions in defn must be explicitly given as arguments.
      For example, this is not valid:

          defn increment_by_y_while_less_than_10(y) do
            while x = 0, Nx.less(x, 10) do
              x + y
            end
          end

      In the example above, we want to increment "x" by "y" while it is less than 10. \
      However, the code won't compile because "y" is used inside "while" but not \
      explicitly defined as part of "while". You must fix it like so:

          defn increment_by_y_while_less_than_10(y) do
            while {x = 0, y}, Nx.less(x, 10) do
              {x + y, y}
            end
          end

      """
    end

    context || acc
  end

  ## Constant helpers and related optimizations

  defp constant(%{vectorized_axes: [_ | _]} = out, number) do
    tensor(Nx.fill(out, number, type: out.type))
  end

  defp constant(%{shape: shape, type: type} = out, number) do
    number =
      cond do
        is_integer(number) and Nx.Type.float?(type) ->
          Complex.multiply(1.0, number)

        not is_integer(number) and Nx.Type.integer?(type) ->
          raise ArgumentError,
                "value #{inspect(number)} is not valid for constant of type #{inspect(type)}"

        number ->
          number
      end

    %{out | data: constant_expr(shape, type, number)}
  end

  defp constant_expr(shape, type, number) do
    %Expr{id: {number, type, shape}, op: :constant, args: [number], context: nil}
  end

  defp constant_binary(tensor, c) do
    Nx.BinaryBackend.constant(%T{type: tensor.type, names: [], shape: {}}, c, [])
  end

  defp maybe_constant(expr) do
    case expr do
      %T{data: %Expr{op: :constant, args: [number]}} -> number
      _ -> nil
    end
  end

  defp ensure_compatible(t, out) do
    t
    |> Nx.as_type(out.type)
    |> Nx.broadcast(out.shape)
    |> Map.replace!(:names, out.names)
  end

  # Rewrite commutative operations so the constant always come on the left
  defp commute(out, context, op, fun, s1, t1, t2) do
    {a1, a2} =
      case t2 do
        %T{data: %Expr{op: ^op, args: [%T{data: %Expr{op: :constant, args: [s2]}}, t3]}} ->
          nullary_out = %{out | shape: {}, names: []}

          if s1 do
            {constant(nullary_out, fun.(s1, s2)), t3 |> Nx.broadcast(out.shape)}
          else
            {constant(nullary_out, s2), apply(Nx, op, [t1, t3]) |> Nx.broadcast(out.shape)}
          end

        %T{} ->
          case t1 do
            %T{data: %Expr{op: ^op, args: [%T{data: %Expr{op: :constant, args: [s1]}}, t3]}} ->
              nullary_out = %{out | shape: {}, names: []}
              {constant(nullary_out, s1), apply(Nx, op, [t2, t3]) |> Nx.broadcast(out.shape)}

            %T{} ->
              {t1, t2}
          end
      end

    binary_expr(out, context, op, a1, a2)
  end

  defp binary_expr(out, context, op, arg1, arg2) do
    c1 = maybe_constant(arg1)
    c2 = maybe_constant(arg2)

    if c1 && c2 do
      apply(Nx.BinaryBackend, op, [
        %{out | shape: {}, names: []},
        constant_binary(arg1, c1),
        constant_binary(arg2, c2)
      ])
      |> Nx.to_number()
      |> then(&constant(out, &1))
    else
      expr(out, context, op, [arg1, arg2])
    end
  end

  defp unary_expr(out, context, op, arg) do
    if c = maybe_constant(arg) do
      apply(Nx.BinaryBackend, op, [%{out | shape: {}, names: []}, constant_binary(arg, c)])
      |> Nx.to_number()
      |> then(&constant(out, &1))
    else
      expr(out, context, op, [arg])
    end
  end

  ## Inspect

  import Inspect.Algebra, except: [to_doc: 2]

  @impl true
  # Special case for constants since we show them inline in regular printing
  def inspect(%T{data: %Expr{op: :constant, args: [constant]}}, opts) do
    concat([line(), color("Nx.Defn.Expr", :map, opts), line(), to_string(constant)])
  end

  def inspect(tensor, opts) do
    {_, %{exprs: exprs, parameters: parameters, length: length}} =
      recur_inspect(tensor, %{
        cache: %{},
        exprs: [],
        parameters: [],
        opts: opts,
        length: 0,
        limit: opts.limit
      })

    concat(line(), color("Nx.Defn.Expr", :map, opts))
    |> append_lines(parameters, length + 3)
    |> append_lines(exprs, length + 3)
  end

  defp recur_inspect(%T{data: %Expr{op: :constant, args: [value]}}, state) do
    {to_string(value), state}
  end

  defp recur_inspect(%T{data: %Expr{op: :fun, args: [_, _, {m, f, a}]}}, state) do
    {[?&, Exception.format_mfa(m, f, a)], state}
  end

  defp recur_inspect(%T{data: %Expr{op: :metadata, args: [tensor, metadata]}}, state)
       when not is_map_key(metadata, :inspect) do
    recur_inspect(tensor, state)
  end

  defp recur_inspect(%T{data: %Expr{op: :optional, args: [expr, _default, _callback]}}, state) do
    recur_inspect(expr, state)
  end

  defp recur_inspect(%T{data: %Expr{id: id, op: op, args: args}} = tensor, state) do
    %{limit: limit, cache: cache} = state

    case cache do
      %{^id => var_name} ->
        {var_name, state}

      %{} when limit == 0 ->
        state = state |> decrement_limit(limit) |> store_line(:parameters, "...", "")
        var_name = var_name(state)
        {var_name, put_in(state.cache[id], var_name)}

      %{} when limit < 0 ->
        var_name = var_name(state)
        {var_name, put_in(state.cache[id], var_name)}

      %{} ->
        {var_name, state} =
          cached_recur_inspect(op, args, to_type_shape(tensor), decrement_limit(state, limit))

        {var_name, put_in(state.cache[id], var_name)}
    end
  end

  defp recur_inspect(tuple, state) when is_tuple(tuple) do
    {args, state} =
      tuple
      |> Tuple.to_list()
      |> Enum.map_reduce(state, &recur_inspect/2)

    {[?{, Enum.intersperse(args, ", "), ?}], state}
  end

  defp recur_inspect(list, %{opts: opts} = state) when is_list(list) do
    if list != [] and Keyword.keyword?(list) do
      inspect =
        Enum.map_intersperse(list, ", ", fn {key, value} ->
          [Macro.inspect_atom(:key, key), " ", doc_inspect(value, opts)]
        end)

      {inspect, state}
    else
      {args, state} = Enum.map_reduce(list, state, &recur_inspect/2)
      {[?[, Enum.intersperse(args, ", "), ?]], state}
    end
  end

  defp recur_inspect(term, state) do
    {doc_inspect(term, state.opts), state}
  end

  defp cached_recur_inspect(:parameter, [pos], type_shape, state) when is_integer(pos) do
    var_name = var_name(state)
    parameter = "parameter #{var_name}:#{pos}"
    {var_name, store_line(state, :parameters, parameter, type_shape)}
  end

  defp cached_recur_inspect(:tensor, [_], type_shape, state) do
    var_name = var_name(state)
    parameter = "tensor #{var_name}"
    {var_name, store_line(state, :parameters, parameter, type_shape)}
  end

  defp cached_recur_inspect(op, args, type_shape, state) do
    {args, state} = traverse_args(op, args, state)
    var_name = var_name(state)
    expr = IO.iodata_to_binary(["#{var_name} = #{op} " | Enum.intersperse(args, ", ")])
    {var_name, store_line(state, :exprs, expr, type_shape)}
  end

  defp traverse_args(:token, [%Nx.Defn.Token{hooks: hooks}], state) do
    Enum.map_reduce(hooks, state, fn %{name: name, expr: expr}, state ->
      {expr, state} = recur_inspect(expr, state)
      {[Macro.inspect_atom(:key, name), " ", expr], state}
    end)
  end

  defp traverse_args(:cond, [clauses, other], state) do
    Enum.map_reduce(clauses ++ [{true, other}], state, fn {condition, body}, state ->
      {condition, state} = recur_inspect(condition, state)
      {body, state} = recur_inspect(body, state)
      {[condition, " -> ", body], state}
    end)
  end

  defp traverse_args(:while, [initial, _arg, _condition, _body], state),
    do: traverse_args([initial], state)

  defp traverse_args(:metadata, [tensor, %{inspect: inspect}], state),
    do: traverse_args([tensor, inspect], state)

  defp traverse_args(_op, [tuple | args], state) when is_tuple(tuple),
    do: traverse_args(args, state)

  defp traverse_args(_op, args, state),
    do: traverse_args(args, state)

  defp traverse_args(args, state) do
    Enum.map_reduce(args, state, &recur_inspect/2)
  end

  defp decrement_limit(state, :infinity), do: state
  defp decrement_limit(state, limit), do: %{state | limit: limit - 1}

  defp doc_inspect(term, opts) do
    term
    |> Inspect.Algebra.to_doc(opts)
    |> Inspect.Algebra.format(:infinity)
  end

  defp append_lines(doc, lines, length) do
    List.foldr(lines, doc, fn {line, type_shape}, acc ->
      concat([acc, line(), line, String.duplicate(" ", length - byte_size(line)), type_shape])
    end)
  end

  defp store_line(%{length: length} = state, key, line, type_shape) do
    tail = Map.fetch!(state, key)
    %{state | key => [{line, type_shape} | tail], length: max(length, byte_size(line))}
  end

  defp var_name(%{cache: cache}) do
    IO.iodata_to_binary(counter_to_name(map_size(cache)))
  end

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [?a + counter]

  defp to_type_shape(%{vectorized_axes: [], type: type, shape: shape}) do
    brackets =
      shape
      |> Tuple.to_list()
      |> Enum.map(&[?[, Integer.to_string(&1), ?]])

    IO.iodata_to_binary([Nx.Type.to_string(type) | brackets])
  end
end
