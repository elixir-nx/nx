target = System.get_env("EXLA_TARGET", "host")
ExUnit.start(exclude: :platform, include: [platform: String.to_atom(target)])

defmodule ExlaHelpers do
  def client(), do: Exla.Client.fetch!(:default)
end
