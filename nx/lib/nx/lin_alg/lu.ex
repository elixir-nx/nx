defmodule Nx.LinAlg.LU do
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

    {p, a_prime} = lu_validate_and_pivot(a)

    {n, _} = Nx.shape(a)

    l = u = Nx.fill(a_prime, 0.0)

    {l, u, _} =
      while {l, u, {a_prime, eps, n}}, j <- 0..(n - 1) do
        l = Nx.put_slice(l, [j, j], Nx.tensor([[1.0]]))

        {u, _} =
          while {u, {l, a_prime, eps, j, i = 0}}, Nx.less_equal(i, j) do
            sum = vector_dot_slice(u[[.., j]], l[i], i)
            a_ij = a_prime[i][j]

            value = a_ij - sum

            if Nx.less(Nx.abs(value), eps) do
              {Nx.put_slice(u, [i, j], Nx.tensor([[0.0]])), {l, a_prime, eps, j, i + 1}}
            else
              {Nx.put_slice(u, [i, j], Nx.reshape(value, {1, 1})), {l, a_prime, eps, j, i + 1}}
            end
          end

        {l, _} =
          while {l, {u, a_prime, eps, j, n, i = j + 1}}, Nx.less_equal(i, n - 1) do
            sum = vector_dot_slice(u[[.., j]], l[i], i)

            a_ij = a_prime[i][j]
            u_jj = u[j][j]

            value =
              cond do
                u_jj != 0 ->
                  (a_ij - sum) / u_jj

                a_ij >= sum ->
                  Nx.Constants.infinity()

                true ->
                  Nx.Constants.neg_infinity()
              end

            if Nx.abs(value) < eps do
              {Nx.put_slice(l, [i, j], Nx.tensor([[0]])), {u, a_prime, eps, j, n, i + 1}}
            else
              {Nx.put_slice(l, [i, j], Nx.reshape(value, {1, 1})), {u, a_prime, eps, j, n, i + 1}}
            end
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
    u = Nx.select(Nx.iota({n}) < last_idx, u, 0)
    {n} = Nx.shape(v)
    v = Nx.select(Nx.iota({n}) < last_idx, v, 0)
    Nx.dot(u, v)
  end

  defnp lu_validate_and_pivot(t) do
    {n, _} = Nx.shape(t)
    p = Nx.iota({n})

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

    permutation = Nx.new_axis(p, 1) == Nx.iota({1, n})

    {permutation, t[p]}
  end

  defn lu_grad({l, u}, {dl, du}) do
    # Definition taken from https://arxiv.org/pdf/2009.10071.pdf
    # Equation (3)
    r_inv = Nx.LinAlg.invert(u)

    m = Nx.dot(u, Nx.LinAlg.adjoint(du)) |> Nx.subtract(Nx.dot(Nx.LinAlg.adjoint(dl), l))

    # copyltu
    m_ltu = Nx.tril(m) |> Nx.add(m |> Nx.tril(k: -1) |> Nx.LinAlg.adjoint())

    da = dl |> Nx.add(Nx.dot(l, m_ltu)) |> Nx.dot(Nx.LinAlg.adjoint(r_inv))

    [da]
  end
end
