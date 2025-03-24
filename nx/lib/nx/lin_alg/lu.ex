defmodule Nx.LinAlg.LU do
  @moduledoc false
  import Nx.Defn

  defn lu(a, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-10)

    vectorized_axes = a.vectorized_axes

    result =
      a
      |> Nx.revectorize([collapsed_axes: :auto],
        target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
      )
      |> lu_matrix(opts)
      |> revectorize_result(a.shape, vectorized_axes)

    custom_grad(result, [a], fn g ->
      lu_grad(result, g)
    end)
  end

  defnp lu_matrix(a, opts \\ []) do
    eps = opts[:eps]
    type = Nx.Type.to_floating(a.type)
    real_type = Nx.Type.to_real(type)

    {p, a_prime} = lu_validate_and_pivot(a)
    # {p, a_prime} = {Nx.eye(a.shape, vectorized_axes: a.vectorized_axes, type: a.type), a}
    a_prime = Nx.as_type(a_prime, type)

    {n, _} = Nx.shape(a)

    l = u = Nx.fill(a_prime, 0.0)
    [eps, _] = Nx.broadcast_vectors([Nx.as_type(eps, real_type), l])

    {l, u, _} =
      while {l, u, {a_prime, eps, n}}, j <- 0..(n - 1) do
        l = Nx.put_slice(l, [j, j], Nx.tensor([[1.0]], type: type))
        [j, i, _] = Nx.broadcast_vectors([j, 0, l])

        {u, _} =
          while {u, {l, a_prime, eps, j, i}}, i <= j do
            sum = vector_dot_slice(u[[.., j]], l[i], i)
            a_ij = a_prime[i][j]

            value = a_ij - sum

            updated_u =
              if Nx.abs(value) < eps do
                Nx.put_slice(u, [i, j], Nx.tensor([[0]], type: type))
              else
                Nx.put_slice(u, [i, j], Nx.reshape(value, {1, 1}))
              end

            {updated_u, {l, a_prime, eps, j, i + 1}}
          end

        {l, _} =
          while {l, {u, a_prime, eps, j, n, i = j + 1}}, i <= n - 1 do
            sum = vector_dot_slice(u[[.., j]], l[i], i)

            a_ij = a_prime[i][j]
            u_jj = u[j][j]

            value =
              case Nx.Type.complex?(type) do
                true ->
                  if u_jj != 0 do
                    (a_ij - sum) / u_jj
                  else
                    Nx.Constants.nan(real_type)
                  end

                false ->
                  cond do
                    u_jj != 0 ->
                      (a_ij - sum) / u_jj

                    a_ij >= sum ->
                      Nx.Constants.infinity(real_type)

                    true ->
                      Nx.Constants.neg_infinity(real_type)
                  end
              end

            updated_l =
              if Nx.abs(value) < eps do
                Nx.put_slice(l, [i, j], Nx.tensor([[0]], type: type))
              else
                Nx.put_slice(l, [i, j], Nx.reshape(value, {1, 1}))
              end

            {updated_l, {u, a_prime, eps, j, n, i + 1}}
          end

        {l, u, {a_prime, eps, n}}
      end

    {p, l, u}
  end

  deftransformp revectorize_result({p, l, u}, shape, vectorized_axes) do
    {p_shape, l_shape, u_shape} = Nx.Shape.lu(shape)

    {
      Nx.revectorize(p, vectorized_axes, target_shape: p_shape),
      Nx.revectorize(l, vectorized_axes, target_shape: l_shape),
      Nx.revectorize(u, vectorized_axes, target_shape: u_shape)
    }
  end

  defnp vector_dot_slice(u, v, last_idx) do
    {n} = Nx.shape(u)
    selector = Nx.iota({n}) < last_idx
    u = Nx.select(selector, u, 0)
    v = Nx.select(selector, v, 0)
    Nx.dot(u, v)
  end

  defnp lu_validate_and_pivot(t) do
    {n, _} = Nx.shape(t)
    p = Nx.iota({n}, vectorized_axes: t.vectorized_axes)

    {p, _} =
      while {p, t}, i <- 0..(n - 2) do
        max_idx =
          Nx.select(Nx.iota({n}) < i, 0, Nx.abs(t[[.., i]]))
          |> Nx.argmax(axis: 0)

        if max_idx == i do
          {p, t}
        else
          indices = Nx.stack([i, max_idx]) |> Nx.reshape({2, 1})
          updates = Nx.stack([p[max_idx], p[i]])

          p = Nx.indexed_put(p, indices, updates)

          {p, Nx.take(t, p)}
        end
      end

    # The comparison order here is deliberate, because if
    # we use p == iota instead, we get the inverse/transposed permutation.
    permutation = Nx.iota({n, 1}) == Nx.new_axis(p, 0)

    {Nx.as_type(permutation, t.type), t[p]}
  end

  defn lu_grad({p, l, u}, {_dp, dl, du}) do
    # Definition taken from https://arxiv.org/pdf/2009.10071.pdf
    # Equation (3)

    u_h = Nx.LinAlg.adjoint(u)
    l_h = Nx.LinAlg.adjoint(l)
    p_t = Nx.LinAlg.adjoint(p)

    lh_dl = Nx.dot(l_h, dl)
    du_uh = Nx.dot(du, u_h)

    lt_inv = Nx.LinAlg.invert(l_h)
    ut_inv = Nx.LinAlg.invert(u_h)

    df = lh_dl |> Nx.tril(k: -1) |> Nx.add(Nx.triu(du_uh))
    da = p_t |> Nx.dot(lt_inv) |> Nx.dot(df) |> Nx.dot(ut_inv)

    [da]
  end
end
