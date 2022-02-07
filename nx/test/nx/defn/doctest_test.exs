defmodule Nx.DoctestTest do
  use ExUnit.Case, async: true

  doctest Nx, except: [sigil_M: 2, sigil_V: 2]
end
