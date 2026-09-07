# AGENTS.md

Guidance for AI agents (and human contributors) working in this repository.
This is a monorepo with three umbrella-style projects, each with its own
`mix.exs`, tests, and CHANGELOG:

  * `nx/` - the core `Nx` library (tensors, `defn`, autograd, `Nx.Backend`)
  * `exla/` - the XLA-backed compiler/backend
  * `torchx/` - the LibTorch-backed backend

Run commands from inside the relevant project directory (e.g. `cd nx && mix test`),
not from the repo root.

## Before you open a PR

  * **Keep the diff to one concern.** Reviewers will ask contributors to
    split a PR that mixes a bug fix with a refactor into separate PRs.
    If your change touches unrelated code, remove that part.
  * **Rebase before you push.** Several PRs carried stray, unrelated diffs
    because the contributor's `main` had drifted from upstream. Rebase onto
    `elixir-nx/nx@main` before opening or updating a PR.
  * **Write plain prose in docs and comments.** Maintainers push back hard on
    docs/comments that read as AI-generated filler ("we do this not via X, but
    via Y", unnecessary hedging, restating the obvious). Say what the code
    does, once, in a normal sentence.
  * **Don't reference issue or PR numbers in code comments.** Git history
    already tracks that; explain the *why* in the comment itself instead.
* **Don't use auto-generated text for issues** Issues should be human-readable
    instead of AI dialect. They should contain a description of the problem or
    feature, examples of what is asked or how to reproduce the bug. If the problem
    is straight-forward, prioritize opening PRs directly.

## Code

  * **Placement: Nx core vs. backend vs. compiler.** `Nx` itself should not
    encode behavior specific to one backend or one compiler. If a feature
    needs to be shared by EXLA/Torchx/EMLX or other libraries, it
    may belong in `Nx` or `Nx.Defn.Compiler`.
  * **Don't leak implementation details into public APIs or user docs.**
    Terms like StableHLO or MLIR are EXLA implementation details; they
    shouldn't surface in `Nx` docs or option names.
  * **Use plain, readable variable names**, not one-letter transcriptions of a
    math formula. If a short name doubles as domain terminology (e.g. `gx`
    could mean "gradient of x"), pick something else.
  * **Favor the simplest mechanism that works.** Reviewers regularly ask for
    extra processes, protocols, default arguments, or duplicate loops to be
    removed in favor of a plainer approach.
  * When writing NIFs (`exla`), favor Fine APIs instead of enif_* functions.

## Tests

  * **Compare tensors with `Nx.Testing.assert_equal/2` or `assert_all_close/3`**
    (also available as test helpers in `nx/test/support/helpers.ex`), not raw
    `==` or manual list comparisons; tensor equality needs float tolerance.
  * **Add new test cases to the existing test file and `describe` block**
    for that function, rather than a new module.
  * **Cover vectorized/edge-case inputs**: mixed vectorized axes,
    container arguments, boolean-output ops, and (for `grad`) a non-linear compound
    function, not only the identity case.
  * **Write a regression test that fails without your fix**, especially for
    bug fixes, so the test actually exercises the reported bug.
  * **Fix precision issues at the specific test that needs it**, not by
    widening tolerance repo-wide or skipping a whole suite on one GPU/backend.
  * Don't store large or shared tensors as module attributes in test files;
    build them in `setup` or pass them as `defn` options.

## Docs

  * Use the standard `iex>` doctest form (input line, then result line) for
    examples, not comments describing what a line would return.
  * For new non-trivial machinery (a new compiler pass, a new subsystem),
    include a short overview of what problem it solves before diving into
    the API, not just function docs.
