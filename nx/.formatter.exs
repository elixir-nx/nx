# Used by "mix format"

locals_without_parens = [
  match_shape: 2,
  defn: 2,
  defnp: 2,
  while: 3
]

[
  locals_without_parens: locals_without_parens,
  export: [locals_without_parens: locals_without_parens],
  inputs: ["{mix,.formatter}.exs", "{lib,test}/**/*.{ex,exs}"]
]
