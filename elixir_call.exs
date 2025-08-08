Mix.install([{:exla, path: "exla"}, {:pythonx, "~> 0.4"}])

Pythonx.uv_init("""
[project]
name = "project"
version = "0.0.0"
requires-python = "==3.13.*"
dependencies = [
  "numpy==2.2.2"
]
""")

Nx.global_default_backend(EXLA.Backend)
t = Nx.iota({10})

elixir_fun = fn t, opts ->
  input = Nx.to_flat_list(t)

  {res, _ctx} =
    Pythonx.eval(
      """
      import numpy as np
      arr = np.array(input)

      c = np.cos(arr) + offset

      list(c)
      """,
      %{"input" => input, "offset" => opts[:value]}
    )

  Nx.tensor(Pythonx.decode(res))
end

jit_fun = fn t ->
  s = Nx.size(t)

  out =
    Nx.Shared.elixir_call(%{t | type: Nx.Type.to_floating(t.type)}, [t, [value: 10]], elixir_fun)

  Nx.negate(out)
end

dbg(Nx.Defn.jit_apply(jit_fun, [t]))
