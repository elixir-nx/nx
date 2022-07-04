# Used by "mix format"
[
  import_deps: [:nx],
  locals_without_parens: [deftensor: 1, defdevice: 1, defvalue: 1],
  inputs: ["{mix,.formatter}.exs", "{bench,examples,config,lib,test}/**/*.{ex,exs}"]
]
