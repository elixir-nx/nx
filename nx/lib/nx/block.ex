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

defmodule Nx.Block do
  @moduledoc false

  def name(%{__struct__: module}) do
    case module do
      Nx.Block.LogicalNot -> :logical_not
      Nx.Block.Phase -> :phase
      Nx.Block.AllClose -> :all_close
      Nx.Block.CumulativeSum -> :cumulative_sum
      Nx.Block.CumulativeProduct -> :cumulative_product
      Nx.Block.CumulativeMin -> :cumulative_min
      Nx.Block.CumulativeMax -> :cumulative_max
      Nx.Block.Cholesky -> :cholesky
      Nx.Block.Solve -> :solve
      Nx.Block.QR -> :qr
      Nx.Block.Eigh -> :eigh
      Nx.Block.SVD -> :svd
      Nx.Block.LU -> :lu
      Nx.Block.Determinant -> :determinant
      Nx.Block.Take -> :take
      Nx.Block.TakeAlongAxis -> :take_along_axis
      Nx.Block.TopK -> :top_k
      Nx.Block.FFT2 -> :fft2
      Nx.Block.IFFT2 -> :ifft2
    end
  end
end
