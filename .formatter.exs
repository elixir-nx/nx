# Used by "mix format"

locals_without_parens = [
  defn: 2,
  defnp: 2
]

[
  locals_without_parens: locals_without_parens,
  export: [locals_without_parens: locals_without_parens],
  inputs: ["{mix,.formatter}.exs", "{bench,examples,config,lib,test}/**/*.{ex,exs}"]
]
