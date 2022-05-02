# Changelog

## v0.2.0 (2022-04-28)

This version requires Elixir v1.13+.

  * [Nx] Support atom notation as the type option throughout the API (for example, `:u8`, `:f64`, etc)
  * [Nx] Add support for complex numbers (c64, c128)
  * [Nx] Add `Nx.cumulative_sum/2`, `Nx.cumulative_product/2`, `Nx.cumulative_min/2`, `Nx.cumulative_max/2`
  * [Nx] Add `Nx.conjugate/1`, `Nx.phase/1`, `Nx.real/1`, and `Nx.imag/1`
  * [Nx] Initial support for NaN and Infinity
  * [Nx] Add `:axis` option to `Nx.shuffle/2`
  * [Nx] Add `Nx.axis_index/2`
  * [Nx] Add `Nx.variance/2` to `Nx.standard_deviation/2`
  * [Nx] Rename `Nx.slice_axis/3` to `Nx.slice_along_axis/4`
  * [Nx.Backend] Add support for optional backends
  * [Nx.Constants] Provide a convenient module to host constants
  * [Nx.Defn] Improve error messages throughout the compiler

## v0.1.0 (2022-01-06)

First release.
