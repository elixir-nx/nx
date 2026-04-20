defmodule Nx.Block.LogicalNot do
  defstruct []
end

defmodule Nx.Block.Phase do
  defstruct []
end

defmodule Nx.Block.LinAlg.Cholesky do
  defstruct []
end

defmodule Nx.Block.LinAlg.Solve do
  defstruct []
end

defmodule Nx.Block.LinAlg.QR do
  defstruct eps: 1.0e-10, mode: :reduced
end

defmodule Nx.Block.LinAlg.Eigh do
  defstruct max_iter: 1000, eps: 1.0e-4
end

defmodule Nx.Block.LinAlg.SVD do
  defstruct max_iter: 100, full_matrices?: true
end

defmodule Nx.Block.LinAlg.LU do
  defstruct eps: 1.0e-10
end

defmodule Nx.Block.LinAlg.Determinant do
  defstruct []
end

defmodule Nx.Block.AllClose do
  defstruct equal_nan: false, rtol: 1.0e-5, atol: 1.0e-8
end

defmodule Nx.Block.CumulativeSum do
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.CumulativeProduct do
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.CumulativeMin do
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.CumulativeMax do
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.Take do
  defstruct axis: 0
end

defmodule Nx.Block.TakeAlongAxis do
  defstruct axis: 0
end

defmodule Nx.Block.TopK do
  defstruct k: 1
end

defmodule Nx.Block.FFT2 do
  defstruct eps: nil, lengths: nil, axes: nil
end

defmodule Nx.Block.IFFT2 do
  defstruct eps: nil, lengths: nil, axes: nil
end

defmodule Nx.Block.RFFT do
  defstruct eps: nil, length: nil, axis: nil
end

defmodule Nx.Block.IRFFT do
  defstruct eps: nil, length: nil, axis: nil
end
