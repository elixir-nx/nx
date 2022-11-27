wide_matrix = Nx.tensor([[1, 1, 3, 5], [4, 5, 4, 4], [4, 6, 9, 3]], backend: EXLA.Backend)
matrix = Nx.tensor([[1, 1], [3, 4]])
pinv = Nx.LinAlg.pinv(matrix)
IO.inspect(pinv)
