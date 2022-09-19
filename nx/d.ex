x = Nx.tensor([[1, 1, 3, 5], [4, 5, 4, 4], [4, 6, 9, 3]])
pinv = Nx.LinAlg.pinv(x)
IO.inspect(pinv)
