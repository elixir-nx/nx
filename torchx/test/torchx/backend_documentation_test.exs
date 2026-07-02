defmodule Torchx.BackendDocumentationTest do
  use ExUnit.Case, async: true

  doctest_file "guides/backend_documentation/index.md"
  doctest_file "guides/backend_documentation/nx.md"
  doctest_file "guides/backend_documentation/nx_lin_alg.md"
end
