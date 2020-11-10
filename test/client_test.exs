defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Options.LocalClientOptions

  test "start_link/1 starts up on host device" do
    # Just use the default for now
    assert {:ok, pid} = Client.start_link(%LocalClientOptions{})
  end
end