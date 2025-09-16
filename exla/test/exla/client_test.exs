defmodule EXLA.ClientTest do
  use ExUnit.Case, async: true

  doctest EXLA.Client

  describe "get_supported_platforms/0" do
    test "returns supported platforms with device information" do
      %{host: _} = EXLA.Client.get_supported_platforms()
    end
  end
end
