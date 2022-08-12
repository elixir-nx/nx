defmodule Nx.Defn.Grad do
  @moduledoc false

  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  def transform(to_grad, fun, transform) do
    {to_grad, ids} =
      Composite.traverse(to_grad, %{}, fn to_grad, ids ->
        to_grad = Expr.metadata(to_grad, %{__MODULE__ => :to_grad})
        {to_grad, Map.put(ids, to_grad.data.id, :stop)}
      end)

    # Collect all IDs in the function environment and mark
    # them as stop grads. This is an optimization to avoid
    # traversing trees when not necessary.
    {:env, env} = Function.info(fun, :env)
    ids = stop_grads(env, ids)

    expr = to_grad |> fun.()
    transformed_expr = transform.(expr) |> validate_expr!()
    {parents, nodes} = parents_tree(transformed_expr, ids)

    to_grad_ids = {to_grad, ids}
    grads = %{transformed_expr.data.id => [constant(1.0, transformed_expr)]}

    {graded, _} =
      Composite.traverse(to_grad, {nodes, grads}, &to_grad(&1, to_grad_ids, parents, &2))

    {expr, graded}
  end

  defp constant(float, shape) do
    shape = Nx.shape(shape)
    names = List.duplicate(nil, tuple_size(shape))
    Expr.constant(%T{shape: shape, type: {:f, 32}, names: names}, float, [])
  end

  defp validate_expr!(%T{data: %Expr{}} = expr) do
    expr
  end

  defp validate_expr!(%T{} = t) do
    raise ArgumentError,
          "can only compute gradients of tensor expressions, got: #{inspect(t)}"
  end

  defp validate_expr!(other) do
    validate_expr!(Expr.tensor(other))
  end

  defp stop_grads(list, ids) when is_list(list),
    do: Enum.reduce(list, ids, &stop_grads/2)

  defp stop_grads(tuple, ids) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.reduce(ids, &stop_grads/2)

  defp stop_grads(%T{data: %Expr{id: id}}, ids),
    do: Map.put(ids, id, :stop)

  defp stop_grads(%T{}, ids),
    do: ids

  defp stop_grads(map, ids) when is_map(map),
    do: map |> Map.values() |> Enum.reduce(ids, &stop_grads/2)

  defp stop_grads(_, ids),
    do: ids

  ## Build the parents tree

  @constants [:constant, :tensor, :eye, :iota, :random_uniform, :random_normal] ++
               [:all, :any, :argmax, :argmin] ++
               [:bitwise_and, :bitwise_or, :bitwise_xor, :bitwise_not] ++
               [:logical_and, :logical_or, :logical_xor, :logical_not] ++
               [:left_shift, :right_shift, :count_leading_zeros, :population_count] ++
               [:floor, :round, :ceil, :sign, :is_nan, :is_infinity] ++
               [:equal, :greater, :greater_equal, :less, :less_equal, :not_equal, :argsort]

  defp parents_tree(expr, nodes) do
    Composite.reduce(expr, {%{}, nodes}, &recur_parents_tree/2)
  end

  defp recur_parents_tree(%T{data: %Expr{id: id, op: op}} = t, {parents, nodes}) do
    case nodes do
      %{^id => _} -> {parents, nodes}
      %{} -> parents_args(op, t, id, {parents, Map.put(nodes, id, t)})
    end
  end

  defp parents_args(:metadata, %{data: %{args: [_, %{stop_grad: true}]}}, _id, acc) do
    acc
  end

  # We register cond as a special node to avoid pretraversing it.
  # Instead we traverse it early on on the grad computation.
  defp parents_args(:cond, _, id, {parents, nodes}) do
    {Map.update(parents, __MODULE__, [id], &[id | &1]), nodes}
  end

  defp parents_args(op, t, parent_id, acc) do
    reduce_args(op, t, acc, fn %T{data: %{id: id, op: op}} = arg, {parents, nodes} ->
      if op in @constants do
        {parents, nodes}
      else
        parents = Map.update(parents, id, [parent_id], &[parent_id | &1])
        recur_parents_tree(arg, {parents, nodes})
      end
    end)
  end

  # For some functions, only a subset of the args participate in the grad,
  # so we handle them accordingly here.

  defp reduce_args(:select, %{data: %{args: [_, on_true, on_false | _]}}, acc, fun),
    do: fun.(on_true, fun.(on_false, acc))

  defp reduce_args(:slice, %{data: %{args: [arg | _]}}, acc, fun),
    do: fun.(arg, acc)

  defp reduce_args(:put_slice, %{data: %{args: [arg, _, update | _]}}, acc, fun),
    do: fun.(arg, fun.(update, acc))

  defp reduce_args(:take_along_axis, %{data: %{args: [arg | _]}}, acc, fun),
    do: fun.(arg, acc)

  defp reduce_args(:take, %{data: %{args: [arg | _]}}, acc, fun),
    do: fun.(arg, acc)

  defp reduce_args(:gather, %{data: %{args: [arg | _]}}, acc, fun),
    do: fun.(arg, acc)

  defp reduce_args(:attach_token, %{data: %{args: [_, arg]}}, acc, fun),
    do: fun.(arg, acc)

  defp reduce_args(:while, %{data: %{args: [initial | _]}}, acc, fun),
    do: Composite.reduce(initial, acc, fun)

  defp reduce_args(:optional, %{data: %{args: [_expr, default_impl_expr]}}, acc, fun),
    do: fun.(default_impl_expr, acc)

  defp reduce_args(_op, t, acc, fun),
    do: Tree.apply_args(t, acc, &{&1, fun.(&1, &2)}) |> elem(1)

  ## Recursion

  defp to_grad(arg, to_grad_ids, parents, acc) do
    id = arg.data.id
    acc = traverse_parents(__MODULE__, to_grad_ids, parents, acc)
    acc = traverse_parents(id, to_grad_ids, parents, acc)
    {nodes, grads} = acc

    res = sum_grad(Map.get(grads, id, []))
    {Nx.broadcast(res, arg), {nodes, grads}}
  end

  defp sum_grad([]), do: Expr.tensor(0.0)
  defp sum_grad(gs), do: Enum.reduce(gs, &Nx.add/2)

  defp traverse_parents(id, to_grad_ids, parents, acc) do
    parents
    |> Map.get(id, [])
    |> Enum.reduce(acc, &recur_to_grad(&1, to_grad_ids, parents, &2))
  end

  defp recur_to_grad(id, to_grad_ids, parents, {nodes, grads}) do
    case nodes do
      %{^id => _} ->
        {nodes, grads} = traverse_parents(id, to_grad_ids, parents, {nodes, grads})
        {ans, nodes} = Map.pop!(nodes, id)
        %T{data: %Expr{op: op, args: args}} = ans
        {gs, grads} = Map.pop(grads, id)

        case gs do
          nil ->
            {nodes, grads}

          [_ | _] ->
            g = Enum.reduce(gs, &Nx.add/2)
            {nodes, update_grads(op, args, ans, g, to_grad_ids, grads)}

          _ ->
            g = gs |> Tuple.to_list() |> Enum.map(&sum_grad/1)
            {nodes, update_grads(op, args, ans, g, to_grad_ids, grads)}
        end

      %{} ->
        {nodes, grads}
    end
  end

  defp update_grads(:elem, [%{type: {:tuple, size}} = tuple, pos], _ans, g, _to_grad_ids, grads) do
    update_in(grads[tuple.data.id], fn tuple ->
      tuple = tuple || Tuple.duplicate([], size)
      put_elem(tuple, pos, [g | elem(tuple, pos)])
    end)
  end

  defp update_grads(:while, [initial, arg, condition, body], _ans, gs, _to_grad_ids, grads) do
    gs = List.wrap(gs)
    flatten_initial = Composite.flatten_list([initial])
    context = hd(flatten_initial).data.context
    arg_context = condition.data.context
    gs = Enum.zip_with(gs, flatten_initial, &Nx.broadcast/2)

    # Convert all gradients into while parameters.
    {grad_args, _} =
      Enum.map_reduce(gs, length(gs), fn g, pos ->
        {Expr.parameter(g, arg_context, pos), pos + 1}
      end)

    # Now compute the gradient of the body, first we build the tree as usual.
    {parents, nodes} = parents_tree(body, %{})

    # The bodies have the grad_arg as their gradient, recursively.
    {while_grads, []} =
      Composite.reduce(body, {%{}, grad_args}, fn arg, {grads, [g | gs]} ->
        {Map.put(grads, arg.data.id, [g]), gs}
      end)

    # Now grad over each input.
    {grad_body, _} =
      [arg]
      |> Composite.flatten_list()
      |> Enum.map_reduce({nodes, while_grads}, &to_grad(&1, {arg, %{}}, parents, &2))

    # And finally build a new while.
    {_, while_gs} =
      Expr.while(
        {initial, List.to_tuple(gs)},
        context,
        {arg, List.to_tuple(grad_args)},
        condition,
        {body, List.to_tuple(grad_body)}
      )

    # Now set the computed gradients for each input.
    {grads, []} =
      Enum.reduce(flatten_initial, {grads, Tuple.to_list(while_gs)}, fn arg, {grads, [g | gs]} ->
        {Map.update(grads, arg.data.id, [g], &[g | &1]), gs}
      end)

    grads
  end

  defp update_grads(:cond, [clauses, last], _ans, gs, {to_grad, ids} = to_grad_ids, grads) do
    gs = List.wrap(gs)
    to_grad = Composite.flatten_list([to_grad])

    clauses =
      Enum.map([{true, last} | clauses], fn {head, body} ->
        {parents, nodes} = parents_tree(body, ids)

        {grads, []} =
          Composite.reduce(body, {grads, gs}, fn arg, {grads, [g | gs]} ->
            {Map.put(grads, arg.data.id, [g]), gs}
          end)

        {graded, _} =
          Enum.map_reduce(to_grad, {nodes, grads}, &to_grad(&1, to_grad_ids, parents, &2))

        {head, graded}
      end)

    # Check with grads are non-zero and keep only the ones that are
    used = Enum.map(to_grad, fn _ -> false end)

    used =
      Enum.reduce(clauses, used, fn {_, graded}, used ->
        Enum.zip_with(graded, used, fn expr, flag -> not zero?(expr) or flag end)
      end)

    # Cond may be called even if no input contributes to the grad.
    # So we check it here.
    if true in used do
      [{true, last} | clauses] =
        Enum.map(clauses, fn {head, graded} ->
          {head, graded |> zip_filter(used) |> List.to_tuple()}
        end)

      # Build a new cond expression and assign each derivative to the new grads.
      cond_gs =
        case Expr.cond(clauses, last) do
          res when is_tuple(res) -> Tuple.to_list(res)
          res -> [res]
        end

      {grads, []} =
        to_grad
        |> zip_filter(used)
        |> Enum.reduce({grads, cond_gs}, fn to_grad, {grads, [elem | rest]} ->
          {Map.update(grads, to_grad.data.id, [elem], &[elem | &1]), rest}
        end)

      # We don't replace nodes for cond because the checks are cheap (scalar values)
      # and shared between the original cond and the graded cond.
      grads
    else
      grads
    end
  end

  @reduced_grads [:add, :multiply, :power]
  @verify_grad Application.compile_env(:nx, :verify_grad, false)

  defp update_grads(op, args, ans, g, _to_grad_ids, grads) do
    pairs = grad(op, args, ans, g)

    if @verify_grad do
      count = reduce_args(op, ans, 0, fn _arg, count -> count + 1 end)

      if op not in @reduced_grads and count != length(pairs) do
        raise "ERROR! grad for #{op} returned #{length(pairs)} entries but traversed #{count} entries"
      end
    end

    Enum.reduce(pairs, grads, fn {child, g}, grads ->
      Map.update(grads, child.data.id, [g], &[g | &1])
    end)
  end

  ## Gradients

  defp grad(:metadata, [expr, %{custom_grad: fun}], _ans, g) do
    args = fun.(expr, g)

    unless is_list(args) and Enum.all?(args, &match?({_, _}, &1)) do
      raise "custom_grad/2 must return a list of tuples, " <>
              "where the first element is the expression to continue computing grad " <>
              "and the second element is the updated g"
    end

    args
  end

  defp grad(:metadata, [expr, _], _ans, g) do
    [{expr, g}]
  end

  defp grad(:select, [pred, on_true, on_false], ans, g) do
    d_on_true = Nx.select(pred, g, Expr.tensor(0.0))
    d_on_false = Nx.select(pred, Expr.tensor(0.0), g)
    [unbroadcast(on_true, d_on_true, ans), unbroadcast(on_false, d_on_false, ans)]
  end

  defp grad(:broadcast, [x, shape, axes], _ans, g) do
    [{x, grad_broadcast(x, shape, axes, g)}]
  end

  defp grad(:clip, [operand, min, max], _ans, g) do
    # w.r.t min
    w_min =
      Nx.select(
        Nx.bitwise_and(Nx.greater(min, operand), Nx.less(min, max)),
        Nx.broadcast(g, operand),
        0.0
      )

    # w.r.t operand
    w_operand =
      Nx.select(
        Nx.bitwise_and(Nx.greater(operand, min), Nx.less(operand, max)),
        g,
        0.0
      )

    # w.r.t max
    w_max = Nx.select(Nx.less(max, operand), Nx.broadcast(g, operand), 0.0)

    [
      {operand, Nx.multiply(g, w_operand)},
      {min, Nx.multiply(g, w_min)},
      {max, Nx.multiply(g, w_max)}
    ]
  end

  defp grad(:squeeze, [x, axes], _ans, g) do
    [{x, Nx.broadcast(g, x.shape, axes: Nx.axes(x.shape) -- axes)}]
  end

  defp grad(:reshape, [x], _ans, g) do
    [{x, Nx.reshape(g, x)}]
  end

  defp grad(:transpose, [x, axes], _ans, g) do
    [{x, Nx.transpose(g, axes: argsort(axes))}]
  end

  defp grad(:pad, [x, value, padding_config], _ans, g) do
    inverse_padding_config = Enum.map(padding_config, fn {lo, hi, _} -> {-lo, -hi, 0} end)
    unpadded = Nx.pad(g, 0.0, inverse_padding_config)

    start_indices = List.duplicate(0, Nx.rank(unpadded))
    lengths = Tuple.to_list(unpadded.shape)
    strides = padding_config |> Enum.map(fn {_, _, interior} -> interior + 1 end)

    g_operand = Nx.slice(unpadded, start_indices, lengths, strides: strides)
    g_value = Nx.subtract(Nx.sum(g), Nx.sum(g_operand))

    [{x, g_operand}, {value, g_value}]
  end

  defp grad(:slice, [x, start_indices, _lengths, strides], _ans, g) do
    padding_config = Enum.map(strides, &{0, 0, &1 - 1})
    pad_value = 0.0
    g = Nx.pad(g, pad_value, padding_config)

    zeros = Nx.broadcast(Expr.tensor(0.0), x)
    [{x, Nx.put_slice(zeros, start_indices, g)}]
  end

  defp grad(:put_slice, [x, start_indices, update], _ans, g) do
    zeros = Nx.broadcast(Expr.tensor(0.0), update)

    operand_t = Nx.put_slice(g, start_indices, zeros)
    update_t = Nx.slice(g, start_indices, Tuple.to_list(Nx.shape(update)))

    [{x, operand_t}, {update, update_t}]
  end

  defp grad(:indexed_put, [target, indices, updates], _ans, g) do
    zeros = Nx.broadcast(Expr.tensor(0.0), updates)

    target_g = Nx.indexed_put(g, indices, zeros)
    updates_g = g |> Nx.gather(indices) |> Nx.reshape(updates.shape)
    indices_g = Nx.broadcast(Expr.tensor(0.0), indices)

    [{target, target_g}, {indices, indices_g}, {updates, updates_g}]
  end

  defp grad(:indexed_add, [target, indices, updates], _ans, g) do
    zeros = Nx.broadcast(Expr.tensor(0.0), updates)

    target_g = Nx.indexed_add(g, indices, zeros)
    updates_g = g |> Nx.gather(indices) |> Nx.reshape(updates.shape)
    indices_g = Nx.broadcast(Expr.tensor(0.0), indices)

    [{target, target_g}, {indices, indices_g}, {updates, updates_g}]
  end

  defp grad(:reverse, [x, axes], _ans, g) do
    [{x, Nx.reverse(g, axes: axes)}]
  end

  defp grad(:sum, [x, opts], _ans, g) do
    [{x, reduce_g(x, opts, g)}]
  end

  @reduce_min_max_ops [:reduce_max, :reduce_min]

  defp grad(op, [x, opts], ans, g) when op in @reduce_min_max_ops do
    g = reduce_g(x, opts, g)
    axes = opts[:axes] || Nx.axes(x)

    shape =
      for {d, i} <- Enum.with_index(Tuple.to_list(x.shape)) do
        if i in axes, do: 1, else: d
      end

    locs = Nx.equal(x, Nx.reshape(ans, List.to_tuple(shape)))
    num = Nx.multiply(g, locs)
    den = Nx.sum(locs, axes: axes, keep_axes: true)
    [{x, Nx.divide(num, den)}]
  end

  defp grad(:dot, [x, axes_x, x_batch_axes, y, axes_y, y_batch_axes], ans, g) do
    g = Nx.broadcast(g, ans)

    batch_gx = up_to(0, length(x_batch_axes))
    batch_gy = up_to(0, length(y_batch_axes))

    contract_gx = up_to(Nx.rank(x.shape) - length(axes_x), Nx.rank(g.shape))
    contract_gy = up_to(length(y_batch_axes), Nx.rank(x.shape) - length(axes_x))

    contract_x = (Nx.axes(x.shape) -- axes_x) -- batch_gx
    contract_y = (Nx.axes(y.shape) -- axes_y) -- batch_gy

    transpose_x = Enum.map(argsort(axes_y), &Enum.fetch!(axes_x, &1))
    transpose_y = Enum.map(argsort(axes_x), &Enum.fetch!(axes_y, &1))

    gx =
      g
      |> Nx.dot(contract_gx, batch_gx, y, contract_y, y_batch_axes)
      |> Nx.transpose(axes: argsort(x_batch_axes ++ contract_x ++ transpose_x))

    gy =
      g
      |> Nx.dot(contract_gy, batch_gy, x, contract_x, x_batch_axes)
      |> Nx.transpose(axes: argsort(y_batch_axes ++ contract_y ++ transpose_y))

    [{x, gx}, {y, gy}]
  end

  defp grad(:conv, [x, y, opts], ans, g) do
    grad_conv(x, y, opts, ans, g)
  end

  @window_chooser_op [:window_min, :window_max]

  defp grad(op, [x, window_dimensions, opts], _ans, g) when op in @window_chooser_op do
    padding = opts[:padding]
    strides = opts[:strides]

    fun =
      if op == :window_min,
        do: &Nx.window_scatter_min/5,
        else: &Nx.window_scatter_max/5

    g = fun.(x, g, 0, window_dimensions, padding: padding, strides: strides)
    [{x, g}]
  end

  defp grad(:window_sum, [x, window_dimensions, opts], _, g) do
    strides = opts[:strides]
    window_dilation = opts[:window_dilations]
    base_dilation = List.duplicate(1, Nx.rank(x))
    padding = opts[:padding]

    padding_config =
      conv_lhs_padding(
        x.shape,
        window_dimensions,
        strides,
        g.shape,
        padding,
        base_dilation,
        window_dilation
      )

    padding_config =
      padding_config
      |> Enum.zip(strides)
      |> Enum.map(fn {{lo, hi}, s} -> {lo, hi, s - 1} end)

    g = Nx.pad(g, 0.0, padding_config)

    g =
      Nx.window_sum(
        g,
        window_dimensions,
        strides: base_dilation,
        padding: List.duplicate({0, 0}, Nx.rank(x)),
        window_dilations: window_dilation
      )

    [{x, g}]
  end

  defp grad(:concatenate, [tensors, axis], ans, g) do
    zero_axes = List.duplicate(0, Nx.rank(ans))
    ans_shape_list = Tuple.to_list(ans.shape)

    {pairs, _} =
      Enum.map_reduce(tensors, 0, fn t, limit ->
        t_len = elem(t.shape, axis)
        current_limit = t_len + limit
        start = List.replace_at(zero_axes, axis, limit)
        len = List.replace_at(ans_shape_list, axis, t_len)
        {{t, Nx.slice(g, start, len)}, current_limit}
      end)

    pairs
  end

  defp grad(:cholesky, [input], l, g) do
    num = g |> tril() |> Nx.dot([0], l, [0]) |> Nx.transpose()
    den = l |> Nx.eye() |> Nx.add(1)
    phi_tril = num |> Nx.divide(den) |> tril()

    bm = Nx.LinAlg.triangular_solve(l, phi_tril, transform_a: :transpose)

    dl =
      l
      |> conjugate_if_complex()
      |> Nx.LinAlg.triangular_solve(bm, left_side: false)

    # If we end up supporting the "make_symmetric" option for Nx.LinAlg.cholesky
    # we need to apply: dl := (adjoint(dl) + dl)/2 when the option is true.
    # If the option is applied as Nx.add(tensor, adjoint(tensor)) |> Nx.divide(2)
    # on the expression, no modifications are needed here, because
    # the grad for the transformation is actually the same transformation
    # applied on the grad

    [{input, dl}]
  end

  defp grad(:qr, [{q, r}, input, _opts], ans, [dq, dr]) do
    # Definition taken from https://arxiv.org/pdf/2009.10071.pdf
    # Equation (3)
    {q, r} = Nx.Defn.Expr.tuple(ans, [q, r])
    r_inv = Nx.LinAlg.invert(r)

    m = Nx.dot(r, Nx.LinAlg.adjoint(dr)) |> Nx.subtract(Nx.dot(Nx.LinAlg.adjoint(dq), q))

    # copyltu
    m_ltu = tril(m) |> Nx.add(m |> tril_strict() |> Nx.LinAlg.adjoint())

    da = dq |> Nx.add(Nx.dot(q, m_ltu)) |> Nx.dot(Nx.LinAlg.adjoint(r_inv))

    [{input, da}]
  end

  defp grad(:lu, [{p, l, u}, input, _opts], ans, [_dp, dl, du]) do
    # Definition taken from: https://sethaxen.com/blog/2021/02/differentiating-the-lu-decomposition/
    # Where dF = tril_strict(L^* . dL) + triu(dU . U^*)
    # dA = P^t . (L^*)^-1 . dF . (U^*)^-1

    {p, l, u} = Nx.Defn.Expr.tuple(ans, [p, l, u])

    u_h = Nx.LinAlg.adjoint(u)
    l_h = Nx.LinAlg.adjoint(l)
    p_t = Nx.LinAlg.adjoint(p)

    lh_dl = Nx.dot(l_h, dl)
    du_uh = Nx.dot(du, u_h)

    lt_inv = Nx.LinAlg.invert(l_h)
    ut_inv = Nx.LinAlg.invert(u_h)

    df = lh_dl |> tril_strict() |> Nx.add(triu(du_uh))
    da = p_t |> Nx.dot(lt_inv) |> Nx.dot(df) |> Nx.dot(ut_inv)

    [{input, da}]
  end

  defp grad(:sort, [t, opts], _ans, g) do
    idx = Nx.argsort(t, opts)
    take_along_opts = Keyword.take(opts, [:axis])
    g = Nx.take_along_axis(g, idx, take_along_opts)
    [{t, g}]
  end

  defp grad(:take_along_axis, [t, i, axis], _ans, g) do
    num_elements = i |> Nx.shape() |> Tuple.product()

    # Convert `i`, the take_along_axis indices, to a list of
    # fully qualified (i.e. [0, 2, 1] for a {_, _, _}-shaped tensor)
    # indices

    indices =
      0..(Nx.rank(g) - 1)//1
      |> Enum.map(fn
        # For the axis of interest, we'll use the actual take_along_axis indices
        ^axis ->
          Nx.reshape(i, {num_elements, 1})

        axis ->
          i
          |> Nx.iota(axis: axis)
          |> Nx.reshape({num_elements, 1})
      end)
      |> Nx.concatenate(axis: 1)

    # Since g is produced through the given indices,
    # we can reshape g to be a {num_elements} shaped tensor
    # which will directly correspond to each of the reshaped
    # indices above
    updates = Nx.reshape(g, {num_elements})

    # The intuition for this grad is that for each index taken, we'll
    # add the corresponding result grad to the original
    g =
      t
      |> Expr.broadcast(0, Nx.shape(t), Nx.axes(t))
      |> Nx.indexed_add(indices, updates)

    [{t, g}]
  end

  defp grad(:take, [t, i, axis], _ans, g) do
    axes_range = 0..(Nx.rank(t) - 1)//1

    indices_shape =
      axes_range
      |> Enum.flat_map(fn
        ^axis -> Tuple.to_list(i.shape)
        _ -> [1]
      end)
      |> List.to_tuple()

    idx_tiling =
      t.shape
      |> Tuple.to_list()
      |> Enum.with_index(fn
        _x, ^axis ->
          List.duplicate(1, Nx.rank(i))

        x, _ ->
          x
      end)
      |> List.flatten()

    num_elements = Tuple.product(g.shape)

    indices_for_axis =
      i
      |> Nx.reshape(indices_shape)
      |> Nx.tile(idx_tiling)

    axis_offset = Nx.rank(i) - 1

    indices =
      axes_range
      |> Enum.map(fn
        ^axis ->
          indices_for_axis
          |> Nx.reshape({num_elements, 1})

        current when current < axis ->
          indices_for_axis
          |> Nx.iota(axis: current)
          |> Nx.reshape({num_elements, 1})

        current when current > axis ->
          indices_for_axis
          |> Nx.iota(axis: current + axis_offset)
          |> Nx.reshape({num_elements, 1})
      end)
      |> Nx.concatenate(axis: 1)

    updates = Nx.reshape(g, {num_elements})

    g =
      t
      |> Expr.broadcast(0, Nx.shape(t), Nx.axes(t))
      |> Nx.indexed_add(indices, updates)

    [{t, g}]
  end

  defp grad(:gather, [t, i], _ans, g) do
    rank = Nx.rank(t)
    num_elements = i.shape |> Tuple.product() |> div(rank)

    indices = Nx.reshape(i, {num_elements, rank})
    updates = Nx.reshape(g, {num_elements})

    g = t |> Expr.broadcast(0, t.shape, Nx.axes(t)) |> Nx.indexed_add(indices, updates)
    [{t, g}]
  end

  defp grad(:add, [x, y], ans, g) do
    if x.data.id == y.data.id do
      [{x, Nx.multiply(g, 2.0)}]
    else
      [unbroadcast(x, g, ans), unbroadcast(y, g, ans)]
    end
  end

  defp grad(:subtract, [x, y], ans, g) do
    [unbroadcast(x, g, ans), unbroadcast(y, Nx.negate(g), ans)]
  end

  defp grad(:multiply, [x, y], ans, g) do
    if x.data.id == y.data.id do
      [{x, Nx.multiply(g, Nx.multiply(2.0, x))}]
    else
      [unbroadcast(x, Nx.multiply(g, y), ans), unbroadcast(y, Nx.multiply(g, x), ans)]
    end
  end

  defp grad(:divide, [x, y], ans, g) do
    [
      unbroadcast(x, Nx.divide(g, y), ans),
      unbroadcast(y, Nx.multiply(g, Nx.negate(Nx.divide(ans, y))), ans)
    ]
  end

  defp grad(:remainder, [x, y], ans, g) do
    [
      unbroadcast(x, g, ans),
      unbroadcast(y, Nx.multiply(g, Nx.negate(Nx.floor(Nx.divide(x, y)))), ans)
    ]
  end

  defp grad(:power, [x, y], ans, g) do
    case y do
      %T{data: %Expr{op: :constant, args: [y]}} ->
        exponent = if y == 0.0, do: 1.0, else: y - 1.0
        gx = Nx.multiply(y, Nx.power(x, exponent))
        [unbroadcast(x, Nx.multiply(g, gx), ans)]

      %{} ->
        exponent = Nx.select(Nx.equal(y, 0.0), 1.0, Nx.subtract(y, 1.0))
        base = Nx.select(Nx.equal(x, 0.0), 1.0, x)

        gx = Nx.multiply(y, Nx.power(x, exponent))
        gy = Nx.multiply(Nx.log(base), ans)
        [unbroadcast(x, Nx.multiply(g, gx), ans), unbroadcast(y, Nx.multiply(g, gy), ans)]
    end
  end

  defp grad(:atan2, [x, y], ans, g) do
    den = Nx.add(Nx.multiply(x, x), Nx.multiply(y, y))

    [
      unbroadcast(x, Nx.multiply(g, Nx.divide(y, den)), ans),
      unbroadcast(y, Nx.multiply(g, Nx.negate(Nx.divide(x, den))), ans)
    ]
  end

  defp grad(op, [x, y], ans, g) when op in [:min, :max] do
    lhs =
      Nx.divide(
        Nx.select(Nx.equal(x, ans), 1.0, 0.0),
        Nx.select(Nx.equal(y, ans), 2.0, 1.0)
      )

    rhs =
      Nx.divide(
        Nx.select(Nx.equal(y, ans), 1.0, 0.0),
        Nx.select(Nx.equal(x, ans), 2.0, 1.0)
      )

    [unbroadcast(x, Nx.multiply(g, lhs), ans), unbroadcast(y, Nx.multiply(g, rhs), ans)]
  end

  defp grad(:as_type, [%{type: {:c, _}} = x], %{type: {output_type, _}}, g)
       when output_type != :c do
    # For downcasting complex to float or integer types, `as_type/2`
    # behaves as: `x |> real() |> as_type(output_type)`
    # Therefore, since as_type doesn't have an intrisic grad in itself,
    # the grad for this case should be the same as `real/1`.
    #
    # For reference, the grad for `real/1` just takes the real part of
    # the accumulated grad
    [{x, Nx.real(g)}]
  end

  defp grad(:as_type, [x], _ans, g) do
    [{x, g}]
  end

  defp grad(:bitcast, [x], _ans, g) do
    [{x, g}]
  end

  defp grad(:abs, [%{type: {:c, _}} = z], ans, g) do
    # For the complex variant of abs(z), we can define the forward-mode
    # derivative abs'(z) as follows (for an element-wise function):
    # abs(z)^2 = z.z*
    # 2*abs(z)*abs'(z) = z'.z* + z.(z')* = 2*real(z*.z')
    # abs'(z) = [2*real(z*.z')] / [2*abs(z)]
    # Which is the same as f(z) / (2*abs(z)) where f(z) = d(abs(z)^2)/dz
    # A similar definition can also be found as _abs_jvp_rule in Jax.

    # Furthermore, abs(z) is always real, so conj(abs(z)) = abs(z).
    # This allows us to use the definition at https://juliadiff.org/ChainRulesCore.jl/stable/maths/arrays.html
    # for the abs_squared reverse-mode derivative:
    # dz = re(g).conj(z)/(2.ans) (where . and / are element-wise multiplication and division)
    # Where (2.ans) is the correction factor that appears from our adapted definition.
    # Also note that we use conj(z) instead of z because we're not dealing with a real tensor.

    # The final correction we need to apply is for the edge case where ans[i,j] = 0.
    # In this scenario, the function dz is undefined, but we can work around this
    # by taking inspiration from the real case below. This leads to the conclusion
    # that abs(0) = 0 is the identity function. Having this in mind, we know that
    # real(g) would be a number, but, more importantly, conj(0) = 0, which takes
    # the numerator for our dz definition to 0.
    # Finally, this allows us to replace the 0-elements in `ans` with 1 (or any number, really)
    # taking dz to 0 at those positions.

    mask = Nx.equal(ans, 0)
    ans_no_zero = Nx.select(mask, 1, ans)
    dz = g |> Nx.real() |> Nx.multiply(Nx.conjugate(z)) |> Nx.divide(ans_no_zero)
    [{z, dz}]
  end

  defp grad(:abs, [x], _ans, g) do
    [{x, Nx.select(Nx.greater_equal(x, 0.0), g, Nx.negate(g))}]
  end

  defp grad(:sqrt, [x], ans, g) do
    [{x, Nx.divide(Nx.multiply(g, 0.5), ans)}]
  end

  defp grad(:cbrt, [x], ans, g) do
    [{x, Nx.divide(g, 3 |> Nx.multiply(ans) |> Nx.multiply(ans))}]
  end

  defp grad(:exp, [x], ans, g) do
    [{x, Nx.multiply(g, ans)}]
  end

  defp grad(:expm1, [x], ans, g) do
    [{x, Nx.multiply(g, Nx.add(ans, 1))}]
  end

  defp grad(:log, [x], _ans, g) do
    [{x, Nx.divide(g, x)}]
  end

  defp grad(:log1p, [x], _ans, g) do
    [{x, Nx.divide(g, Nx.add(x, 1))}]
  end

  defp grad(:sigmoid, [x], ans, g) do
    gs =
      x
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(ans)
      |> Nx.multiply(ans)

    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:negate, [x], _ans, g) do
    [{x, Nx.negate(g)}]
  end

  defp grad(:rsqrt, [x], _ans, g) do
    [{x, Nx.multiply(Nx.multiply(g, -0.5), Nx.power(x, -1.5))}]
  end

  defp grad(:sin, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.cos(x))}]
  end

  defp grad(:asin, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.rsqrt(Nx.subtract(1.0, Nx.multiply(x, x))))}]
  end

  defp grad(:sinh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.cosh(x))}]
  end

  defp grad(:asinh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.rsqrt(Nx.add(Nx.multiply(x, x), 1.0)))}]
  end

  defp grad(:acosh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.rsqrt(Nx.subtract(Nx.multiply(x, x), 1.0)))}]
  end

  defp grad(:atanh, [x], _ans, g) do
    [{x, Nx.divide(g, Nx.subtract(1.0, Nx.multiply(x, x)))}]
  end

  defp grad(:cos, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.negate(Nx.sin(x)))}]
  end

  defp grad(:acos, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.negate(Nx.rsqrt(Nx.subtract(1.0, Nx.multiply(x, x)))))}]
  end

  defp grad(:cosh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.sinh(x))}]
  end

  defp grad(:tan, [x], _ans, g) do
    cos = Nx.cos(x)
    [{x, g |> Nx.divide(cos) |> Nx.divide(cos)}]
  end

  defp grad(:atan, [x], _ans, g) do
    [{x, Nx.divide(g, Nx.add(1.0, Nx.multiply(x, x)))}]
  end

  defp grad(:tanh, [x], ans, g) do
    [{x, Nx.multiply(g, Nx.subtract(1.0, Nx.multiply(ans, ans)))}]
  end

  @half_sqrt_pi :math.sqrt(:math.pi()) / 2
  @two_rsqrt_pi 2 / :math.sqrt(:math.pi())

  defp grad(:erf, [x], _ans, g) do
    gs =
      x
      |> Nx.multiply(x)
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(@two_rsqrt_pi)

    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:erfc, [x], _ans, g) do
    gs =
      x
      |> Nx.multiply(x)
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(-@two_rsqrt_pi)

    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:erf_inv, [x], ans, g) do
    gs = Nx.multiply(@half_sqrt_pi, Nx.exp(Nx.multiply(ans, ans)))
    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:attach_token, [_, x], _ans, g) do
    [{x, g}]
  end

  defp grad(:conjugate, [%{type: {type, _}} = t], _ans, g) do
    if type == :c do
      [{t, Nx.conjugate(g)}]
    else
      [{t, Nx.real(g)}]
    end
  end

  defp grad(:real, [t], _ans, g) do
    # real(z) = (z + conj(z))/2
    # real'(z) = (z' + (conj(z))')/2 = (z' + conj(z'))/2 = real(z')
    [{t, Nx.real(g)}]
  end

  defp grad(:imag, [t], _ans, g) do
    # imag(z) = (z - z*) / 2i
    # imag'(z) = z' - z'* / 2i = imag(z')
    [{t, Nx.imag(g)}]
  end

  defp grad(:fft, args, ans, g), do: grad_fft(:fft, args, ans, g)
  defp grad(:ifft, args, ans, g), do: grad_fft(:ifft, args, ans, g)

  defp grad(:triangular_solve, [a_input, b, opts], x_input, g) do
    # We can model the triangular solve function as X = triangular_solve(a, b)
    # where the function itself depends on the options passed.

    # We can ignore in our calculations the 'lower' option because in all cases we are operating on some form of triangular_solve(A, B) === inv(A).B or B.inv(A)
    # This only needs to be taken into account for the result `da`

    # Therefore, we need to account for left_side and transform_a.
    # The transformations are :none, :transpose and :conjugate,
    # all of which can be applied beforehand to the a matrix.

    # This means we can bifurcate the code through the left_side option
    a =
      case opts[:transform_a] do
        :none -> a_input
        :transpose -> Nx.transpose(a_input)
      end

    a_inv_hermitian = Nx.LinAlg.invert(Nx.LinAlg.adjoint(a))

    x =
      case {Nx.shape(x_input), opts[:left_side]} do
        {{n}, true} -> Nx.reshape(x_input, {n, 1})
        {{n}, false} -> Nx.reshape(x_input, {1, n})
        _ -> x_input
      end

    g =
      case {Nx.shape(g), opts[:left_side]} do
        {{n}, true} -> Nx.reshape(g, {n, 1})
        {{n}, false} -> Nx.reshape(g, {1, n})
        _ -> g
      end

    {da, db} =
      if opts[:left_side] do
        # A.X = B -> X = inv(A).B
        # taking the forward-mode derivative from both sides, we reach the expression:
        # dX = -inv(A).dA.X + inv(A).dB
        # then, we can develop the dot operator <X_bar, dX> to obtain A_bar and B_bar,
        # which are the reverse-mode derivatives w.r.t A and B:
        # <X_bar, dX> = <X_bar, -inv(A).dA.X> + <X_bar, inv(A).dB>
        # = <-inv(A^H).X_bar.X^H, dA> + <inv(A^H).X_bar, dB>
        # which means that:
        # A_bar = inv(A^H).X_bar.X^H
        # B_bar = inv(A^H).X_bar
        da = a_inv_hermitian |> Nx.dot(g |> Nx.dot(Nx.LinAlg.adjoint(x))) |> Nx.negate()
        db = Nx.dot(a_inv_hermitian, g)
        {da, db}
      else
        # X.A = B -> X = B.inv(A)
        # taking a similar approach to the branch above, we get
        # A_bar = -X^H.X_bar.inv(A^H)
        # B_bar = X_bar.inv(A^H)
        da = x |> Nx.LinAlg.adjoint() |> Nx.dot(g) |> Nx.dot(a_inv_hermitian) |> Nx.negate()
        db = Nx.dot(g, a_inv_hermitian)
        {da, db}
      end

    da =
      case opts[:transform_a] do
        :none -> da
        :transpose -> Nx.transpose(da)
      end

    da =
      if opts[:lower] do
        tril(da)
      else
        triu(da)
      end

    db =
      case Nx.shape(x_input) do
        {n} -> Nx.reshape(db, {n})
        _ -> db
      end

    [{a_input, da}, {b, db}]
  end

  defp grad(:quotient, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.quotient/2.

    If a floating point computation is acceptable, consider \
    using an implementation of floor division. See the \
    documentation of `Nx.quotient` for more details.
    """
  end

  defp grad(:reduce, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.reduce/4.

    If you are computing the sum, product, or similar, use the \
    appropriate Nx functions instead. If you have a custom usage \
    of reduce, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  defp grad(:window_reduce, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.window_reduce/5.

    If you are computing the sum, max, or similar of a window, use \
    the appropriate Nx functions instead. If you have a custom usage \
    of window_reduce, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  @error [:map, :window_product]

  defp grad(op, args, _, _) when op in @error do
    raise ArgumentError, """
    cannot compute gradient for Nx.#{op}/#{length(args)}.

    Consider using stop_grad/1 to make the gradient equivalent to \
    the identity function or use custom_grad/2 to define a proper \
    gradient implementation
    """
  end

  defp grad(op, args, _, _) do
    raise ArgumentError, """
    gradient not yet implemented for Nx.#{op}/#{length(args)}.

    Please open up an issue so we can implement the missing gradient
    """
  end

  ## Conv

  defp grad_conv(x, y, opts, ans, g) do
    g = Nx.broadcast(g, ans)

    input_permutation = opts[:input_permutation]
    kernel_permutation = opts[:kernel_permutation]
    output_permutation = opts[:output_permutation]
    strides = opts[:strides]
    padding = opts[:padding]
    lhs_dilation = opts[:input_dilation]
    rhs_dilation = opts[:kernel_dilation]
    feature_group_size = opts[:feature_group_size]
    batch_group_size = opts[:batch_group_size]

    [lhs0, lhs1 | lhs_sdim_axes] = input_permutation
    [rhs0, rhs1 | rhs_sdim_axes] = kernel_permutation
    [_, _ | out_sdim_axes] = output_permutation

    t_lhs_permutation = conv_spec_transpose(input_permutation)
    t_rhs_permutation = conv_spec_transpose(kernel_permutation)
    t_out_permutation = conv_spec_transpose(output_permutation)

    lhs_sdims = conv_sdims(x.shape, lhs_sdim_axes)
    rhs_sdims = conv_sdims(y.shape, rhs_sdim_axes)
    out_sdims = conv_sdims(g.shape, out_sdim_axes)

    rhs =
      cond do
        feature_group_size > 1 ->
          y = reshape_axis_out_of(rhs0, feature_group_size, y)
          reshape_axis_into(rhs0, rhs1, y)

        batch_group_size > 1 ->
          y = reshape_axis_out_of(rhs0, batch_group_size, y)
          reshape_axis_into(rhs0, rhs1, y)

        true ->
          y
      end

    lhs_padding =
      conv_lhs_padding(
        lhs_sdims,
        rhs_sdims,
        strides,
        out_sdims,
        padding,
        lhs_dilation,
        rhs_dilation
      )

    rhs_padding =
      conv_rhs_padding(
        lhs_sdims,
        rhs_sdims,
        strides,
        out_sdims,
        padding,
        lhs_dilation,
        rhs_dilation
      )

    lhs_feature_group_size =
      if batch_group_size > 1, do: batch_group_size, else: feature_group_size

    {rhs_feature_group_size, rhs_batch_group_size} =
      cond do
        batch_group_size > 1 ->
          {batch_group_size, 1}

        feature_group_size > 1 ->
          {1, feature_group_size}

        true ->
          {1, 1}
      end

    revd_weights = Nx.reverse(rhs, axes: rhs_sdim_axes)

    gx =
      Nx.conv(g, revd_weights,
        strides: lhs_dilation,
        padding: lhs_padding,
        input_dilation: strides,
        kernel_dilation: rhs_dilation,
        input_permutation: output_permutation,
        kernel_permutation: t_rhs_permutation,
        output_permutation: input_permutation,
        feature_group_size: lhs_feature_group_size,
        batch_group_size: 1
      )

    gx =
      if batch_group_size > 1 do
        gx = reshape_axis_out_of(lhs1, batch_group_size, gx)
        reshape_axis_into(lhs1, lhs0, gx)
      else
        gx
      end

    gy =
      Nx.conv(x, g,
        strides: rhs_dilation,
        padding: rhs_padding,
        input_dilation: lhs_dilation,
        kernel_dilation: strides,
        input_permutation: t_lhs_permutation,
        kernel_permutation: t_out_permutation,
        output_permutation: t_rhs_permutation,
        feature_group_size: rhs_feature_group_size,
        batch_group_size: rhs_batch_group_size
      )

    [{x, gx}, {y, gy}]
  end

  defp conv_spec_transpose([dim0, dim1 | rest]), do: [dim1, dim0 | rest]

  defp conv_sdims(shape, axes) do
    axes
    |> Enum.map(&elem(shape, &1))
    |> List.to_tuple()
  end

  defp conv_lhs_padding(
         lhs_sdims,
         rhs_sdims,
         strides,
         out_sdims,
         padding,
         lhs_dilation,
         rhs_dilation
       ) do
    lhs_dilated_padding_config = Enum.map(lhs_dilation, &{0, 0, &1 - 1})
    rhs_dilated_padding_config = Enum.map(rhs_dilation, &{0, 0, &1 - 1})
    out_dilated_padding_config = Enum.map(strides, &{0, 0, &1 - 1})
    lhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(lhs_sdims, lhs_dilated_padding_config))
    rhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(rhs_sdims, rhs_dilated_padding_config))
    out_dilated_shape = Tuple.to_list(Nx.Shape.pad(out_sdims, out_dilated_padding_config))

    pad_before = Enum.zip_with(rhs_dilated_shape, padding, fn s, {lo, _} -> s - lo - 1 end)

    pad_after =
      [lhs_dilated_shape, rhs_dilated_shape, out_dilated_shape, pad_before]
      |> Enum.zip_with(fn [l, r, o, p] -> l + r - 1 - o - p end)

    Enum.zip(pad_before, pad_after)
  end

  defp conv_rhs_padding(
         lhs_sdims,
         rhs_sdims,
         strides,
         out_sdims,
         padding,
         lhs_dilation,
         rhs_dilation
       ) do
    lhs_dilated_padding_config = Enum.map(lhs_dilation, &{0, 0, &1 - 1})
    rhs_dilated_padding_config = Enum.map(rhs_dilation, &{0, 0, &1 - 1})
    out_dilated_padding_config = Enum.map(strides, &{0, 0, &1 - 1})
    lhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(lhs_sdims, lhs_dilated_padding_config))
    rhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(rhs_sdims, rhs_dilated_padding_config))
    out_dilated_shape = Tuple.to_list(Nx.Shape.pad(out_sdims, out_dilated_padding_config))

    total_in_pad =
      [out_dilated_shape, rhs_dilated_shape, lhs_dilated_shape]
      |> Enum.zip_with(fn [o, r, l] -> o + r - l - 1 end)

    Enum.zip_with(padding, total_in_pad, fn {lo, _}, hi -> {lo, hi - lo} end)
  end

  defp reshape_axis_into(src, dst, x) do
    perm = for i <- 0..(Nx.rank(x.shape) - 1), i != src, do: i
    perm = List.insert_at(perm, dst, src)
    new_shape = Tuple.delete_at(x.shape, src)
    new_val = elem(new_shape, dst) * elem(x.shape, src)
    new_shape = put_elem(new_shape, dst, new_val)
    Nx.reshape(Nx.transpose(x, axes: perm), new_shape)
  end

  defp reshape_axis_out_of(src, size1, x) do
    size2 = div(elem(x.shape, src), size1)
    new_shape = x.shape
    new_shape = put_elem(new_shape, src, size1)
    new_shape = Tuple.insert_at(new_shape, src + 1, size2)
    Nx.reshape(x, new_shape)
  end

  ## General helpers

  defp unbroadcast(%{shape: shape} = x, res, %{shape: shape}), do: {x, res}

  defp unbroadcast(%{shape: shape} = x, res, %{shape: new_shape}) do
    axes = Nx.Shape.broadcast_axes(shape, new_shape)
    {x, grad_broadcast(x, new_shape, axes, res)}
  end

  defp grad_broadcast(x, shape, axes, g) do
    implicit_axes =
      for {a, i} <- Enum.with_index(axes),
          elem(shape, a) != 1 and elem(x.shape, i) == 1,
          do: {a, i}

    {implicit_axes, broadcast_axes} = Enum.unzip(implicit_axes)
    explicit_axes = Nx.axes(shape) -- axes

    g =
      case explicit_axes ++ implicit_axes do
        [] -> g
        sum_axes -> Nx.sum(g, axes: sum_axes)
      end

    case broadcast_axes do
      [] -> g
      _ -> Nx.broadcast(g, x.shape, axes: Nx.axes(x.shape) -- broadcast_axes)
    end
  end

  defp reduce_g(x, opts, g) do
    axes = opts[:axes]
    keep_axes = opts[:keep_axes]

    if keep_axes || !axes do
      Nx.broadcast(g, x)
    else
      axes = Nx.axes(x.shape) -- axes
      Nx.broadcast(g, x, axes: axes)
    end
  end

  defp zero?(%T{data: %{op: :constant, args: [0.0]}}), do: true
  defp zero?(_), do: false

  defp zip_filter([head | tail], [true | mask]), do: [head | zip_filter(tail, mask)]
  defp zip_filter([_ | tail], [false | mask]), do: zip_filter(tail, mask)
  defp zip_filter([], []), do: []

  defp up_to(i, n) when i < n, do: [i | up_to(i + 1, n)]
  defp up_to(_, _), do: []

  defp argsort(list), do: list |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))

  defp tril(t) do
    t
    |> Nx.iota(axis: 0)
    |> Nx.greater_equal(Nx.iota(t, axis: 1))
    |> Nx.select(t, Nx.tensor(0, type: t.type))
  end

  defp tril_strict(t) do
    t
    |> Nx.iota(axis: 0)
    |> Nx.greater(Nx.iota(t, axis: 1))
    |> Nx.select(t, Nx.tensor(0, type: t.type))
  end

  defp triu(t) do
    t
    |> Nx.iota(axis: 0)
    |> Nx.less_equal(Nx.iota(t, axis: 1))
    |> Nx.select(t, Nx.tensor(0, type: t.type))
  end

  defp conjugate_if_complex(%{type: {:c, _}} = t), do: Nx.conjugate(t)
  defp conjugate_if_complex(t), do: t

  defp grad_fft(kind, [t, opts], _ans, g) do
    nfft = opts[:length]

    grad = apply(Nx, kind, [g, opts])

    formatted_grad =
      case elem(t.shape, Nx.rank(t) - 1) do
        size when size > nfft ->
          # This means the tensor is sliced and we need to pad with zeros
          padding = List.duplicate({0, 0, 0}, Nx.rank(t) - 1) ++ [{0, size - nfft, 0}]
          Nx.pad(grad, 0, padding)

        size when size < nfft ->
          # This means the tensor was padded and we need to slice the result back
          Nx.slice(grad, List.duplicate(0, Nx.rank(t)), Tuple.to_list(t.shape))

        _ ->
          grad
      end

    [{t, formatted_grad}]
  end
end
