defmodule Nx.Block.LogicalNot do
  defstruct []
end

defmodule Nx.Block.Phase do
  defstruct []
end

defmodule Nx.Block.Cholesky do
  defstruct []
end

defmodule Nx.Block.Solve do
  defstruct []
end

defmodule Nx.Block.QR do
  @derive {Nx.Container, containers: [], keep: [:eps, :mode]}
  defstruct eps: 1.0e-10, mode: :reduced
end

defmodule Nx.Block.Eigh do
  @derive {Nx.Container, containers: [], keep: [:max_iter, :eps]}
  defstruct max_iter: 1000, eps: 1.0e-4
end

defmodule Nx.Block.SVD do
  @derive {Nx.Container, containers: [], keep: [:max_iter, :full_matrices?]}
  defstruct max_iter: 100, full_matrices?: true
end

defmodule Nx.Block.LU do
  @derive {Nx.Container, containers: [], keep: [:eps]}
  defstruct eps: 1.0e-10
end

defmodule Nx.Block.Determinant do
  defstruct []
end

defmodule Nx.Block.AllClose do
  @derive {Nx.Container, containers: [], keep: [:equal_nan, :rtol, :atol]}
  defstruct equal_nan: false, rtol: 1.0e-5, atol: 1.0e-8
end

defmodule Nx.Block.CumulativeSum do
  @derive {Nx.Container, containers: [], keep: [:axis, :reverse]}
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.CumulativeProduct do
  @derive {Nx.Container, containers: [], keep: [:axis, :reverse]}
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.CumulativeMin do
  @derive {Nx.Container, containers: [], keep: [:axis, :reverse]}
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.CumulativeMax do
  @derive {Nx.Container, containers: [], keep: [:axis, :reverse]}
  defstruct axis: 0, reverse: false
end

defmodule Nx.Block.Take do
  @derive {Nx.Container, containers: [], keep: [:axis]}
  defstruct axis: 0
end

defmodule Nx.Block.TakeAlongAxis do
  @derive {Nx.Container, containers: [], keep: [:axis]}
  defstruct axis: 0
end

defmodule Nx.Block.TopK do
  @derive {Nx.Container, containers: [], keep: [:k]}
  defstruct k: 1
end

defmodule Nx.Block.FFT2 do
  @derive {Nx.Container, containers: [], keep: [:eps, :lengths, :axes]}
  defstruct eps: nil, lengths: nil, axes: nil
end

defmodule Nx.Block.IFFT2 do
  @derive {Nx.Container, containers: [], keep: [:eps, :lengths, :axes]}
  defstruct eps: nil, lengths: nil, axes: nil
end
