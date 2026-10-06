defmodule EXLA.NIFTest do
  use ExUnit.Case, async: true

  test "load error includes the path, architecture, and target" do
    message =
      EXLA.NIF.load_error(
        ~c"/no/such/libexla",
        :load_failed,
        ~c"cannot open shared object file"
      )

    assert message =~ "/no/such/libexla"
    assert message =~ ":load_failed"
    assert message =~ "cannot open shared object file"
    assert message =~ List.to_string(:erlang.system_info(:system_architecture))
    assert message =~ "XLA_TARGET: #{System.get_env("XLA_TARGET") || "unset"}"
  end
end
