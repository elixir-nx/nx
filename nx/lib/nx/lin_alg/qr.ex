defmodule Nx.LinAlg.QR do
  import Nx.Defn

  defn qr(a, opts \\ []) do
    vectorized_axes = a.vectorized_axes

    a
    |> Nx.revectorize([collapsed_axes: :auto],
      target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
    )
    |> qr_matrix()
    |> revectorize_result(a.shape, vectorized_axes, opts)
  end

  deftransformp revectorize_result({q, r}, shape, vectorized_axes, opts) do
    {q_shape, r_shape} = Nx.Shape.qr(shape, opts)

    {
      Nx.revectorize(q, vectorized_axes, target_shape: q_shape),
      Nx.revectorize(r, vectorized_axes, target_shape: r_shape)
    }
  end

  defnp qr_matrix(a) do
    {m, n} = Nx.shape(a)

    type = Nx.Type.to_floating(Nx.type(a))

    base_h = Nx.eye(m, type: type, vectorized_axes: a.vectorized_axes)
    take_column_shape = {Nx.axis_size(a, 0), 1}
    column_iota = Nx.iota({Nx.axis_size(a, 1), 1}, vectorized_axes: a.vectorized_axes)

    {{q, r}, _} =
      while {{q = base_h, r = Nx.as_type(a, type)}, {column_iota}}, i <- 0..(n - 2) do
        x = Nx.take_along_axis(r, Nx.broadcast(i, take_column_shape), axis: 1)
        selector = Nx.less(column_iota, i)
        x = Nx.flatten(Nx.select(selector, 0, x))

        h = householder_reflector(x, i)
        r = Nx.dot(h, r)
        q = Nx.dot(q, h)
        {{q, r}, {column_iota}}
      end

    {q, r}
  end

  defn householder_reflector(x, i) do
    # x is a {n} tensor
    norm_x = Nx.LinAlg.norm(x)
    sign = Nx.select(x[0] >= 0, 1, -1)
    u = x + sign * norm_x * Nx.indexed_put(Nx.broadcast(0, x), Nx.new_axis(i, 0), 1)
    v = u / Nx.LinAlg.norm(u)

    selector = Nx.iota({Nx.size(x)}) |> Nx.greater_equal(i) |> then(&Nx.outer(&1, &1))

    eye = Nx.eye(Nx.size(x))
    Nx.select(selector, eye - 2 * Nx.outer(v, v), eye)
  end
end
