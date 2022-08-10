defmodule T do
  @type t :: integer
end

defmodule M do
  @type custom :: T.t()
  @spec f(x :: T.t(), y :: custom) :: T.t()
  def f(x, y), do: x + y
end
