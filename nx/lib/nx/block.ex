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
  defstruct []
end

defmodule Nx.Block.Eigh do
  defstruct []
end

defmodule Nx.Block.SVD do
  defstruct []
end

defmodule Nx.Block.LU do
  defstruct []
end

defmodule Nx.Block.Determinant do
  defstruct []
end

defmodule Nx.Block.AllClose do
  defstruct []
end

defmodule Nx.Block.CumulativeSum do
  defstruct []
end

defmodule Nx.Block.CumulativeProduct do
  defstruct []
end

defmodule Nx.Block.CumulativeMin do
  defstruct []
end

defmodule Nx.Block.CumulativeMax do
  defstruct []
end

defmodule Nx.Block.Take do
  defstruct []
end

defmodule Nx.Block.TakeAlongAxis do
  defstruct []
end

defmodule Nx.Block.TopK do
  defstruct []
end

defmodule Nx.Block.FFT2 do
  defstruct []
end

defmodule Nx.Block.IFFT2 do
  defstruct []
end
