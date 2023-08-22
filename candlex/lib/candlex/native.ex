defmodule Candlex.Native do
  @moduledoc false

  use Rustler, otp_app: :candlex, crate: "candlex"

  # Rustler will override all the below stub functions with real NIFs
  def from_binary(_binary, _dtype, _shape), do: error()
  def to_binary(_tensor), do: error()

  defp error(), do: :erlang.nif_error(:nif_not_loaded)
end