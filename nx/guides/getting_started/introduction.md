# What is Nx?

Nx is the numerical computing library of Elixir. Since Elixir´s primary numerical datatypes and structures are not optimized for numerical programming, Nx is the fundamental package built to bridge this gap.

[Elixir Nx](https://github.com/elixir-nx/nx) smoothly integrate to typed, multidimensional data implemented on other
platforms (called [tensors](introduction.html#what-are-tensors)). This support extends to the compilers and
libraries that support those tensors. Nx has four primary capabilities:

- In Nx, tensors hold typed data in multiple, named dimensions.
- Numerical definitions, known as `defn`, support custom code with
  tensor-aware operators and functions.
- [Automatic differentiation](https://arxiv.org/abs/1502.05767), also known as
  autograd or autodiff, supports common computational scenarios
  such as machine learning, simulations, curve fitting, and probabilistic models.
- Broadcasting, which is term for element-by-element operations. Most of the Nx operations
  automatically broadcast using an effective algorithm. You can see more on broadcast
  [here.](intro-to-nx.html#broadcasts)

Here's more about each of those capabilities. Nx tensors can hold
unsigned integers (u2, u4, u8, u16, u32, u64),
signed integers (s2, s4, s8, s16, s32, s64),
floats (f32, f64), brain floats (bf16), and complex (c64, c128).
Tensors support backends implemented outside of Elixir, including Google's
Accelerated Linear Algebra (XLA) and LibTorch.

Numerical definitions have compiler support to allow just-in-time compilation
that support specialized processors to speed up numeric computation including
TPUs and GPUs.

## What are Tensors?

In Nx, we express multi-dimensional data using typed tensors. Simply put,
a tensor is a multi-dimensional array with a predetermined shape and
type. To interact with them, Nx relies on tensor-aware operators rather
than `Enum.map/2` and `Enum.reduce/3`.

It allows us to work with the central theme in numerical computing, systems of equations,
which are often expressed and solved with multidimensional arrays.

For example, this is a two dimensional array:

$$
\begin{bmatrix}
  1 & 2 \\\\
  3 & 4
\end{bmatrix}
$$

As elixir programmers, we can typically express a similar data structure using a list of lists,
like this:

```elixir
[
  [1, 2],
  [3, 4]
]
```

This data structure works fine within many functional programming
algorithms, but breaks down with deep nesting and random access.

On top of that, Elixir numeric types lack optimization for many numerical
applications. They work fine when programs
need hundreds or even thousands of calculations. However, they tend to break
down with traditional STEM applications when a typical problem
needs millions of calculations.

To solve for this, we can simply use Nx tensors, for example:

```elixir
Nx.tensor([[1,2],[3,4]])

Output:
#Nx.Tensor<
s32[2][2]
[
[1, 2],
[3, 4]
]

```

To know Nx, we'll get to know tensors first. The following overview will touch
on the major libraries. Then, future notebooks will take a deep dive into working
with tensors in detail, autograd, and backends. Then, we'll dive into specific
problem spaces like Axon, the machine learning library.
