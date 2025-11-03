# Eigenvalue Decomposition Implementation Notes

## Current Implementation Status

### What Works
The current implementation in `lib/nx/lin_alg/eig.ex` successfully computes:
- **Eigenvalues**: Reliably computed using the QR algorithm with Wilkinson shifts
- **Eigenvectors for well-separated eigenvalues**: Works when eigenvalue gaps are large
- **Balancing**: Pre-conditioning via diagonal similarity transforms (D^-1 * A * D)
- **Hessenberg reduction**: Upper Hessenberg form computed via Householder reflections
- **Schur form**: Quasi-triangular form obtained from shifted QR iterations

### Current Algorithm Pipeline

```
Input Matrix A (n×n)
    ↓
1. Balance: A_bal = D^-1 * A * D
    ↓
2. Hessenberg: A_bal = Q * H * Q^H
    ↓
3. QR Algorithm: H → Schur form S (quasi-upper-triangular)
    ↓
4. Extract Eigenvalues: λ_i from diagonal of S
    ↓
5. Compute Eigenvectors: Inverse iteration on (S - λ_i*I)
    ↓
6. Transform back: V = D * Q * V_schur
    ↓
7. Polish: Refine eigenvectors via inverse iteration on A
    ↓
8. Rayleigh Refinement: Recompute λ_i = v_i^H * A * v_i / ||v_i||^2
    ↓
9. Sort by magnitude (optional)
    ↓
Output: (eigenvalues, eigenvectors)
```

### The Problem: Unreliable Eigenvectors

**Root Cause**: The eigenvector computation (Step 5) uses inverse iteration with normal equations:
```
Solve: (A^H * A + μ*I) * v_new = A^H * v_old
where A = (S - λ_i * I)
```

**Why it fails**:
1. When eigenvalues are close (e.g., λ_1 = 1.0, λ_2 = 0.1), the matrix (S - λ_i*I) is nearly singular for the wrong reasons
2. Inverse iteration can converge to the wrong eigenspace
3. Numerical regularization (μ) prevents convergence to high accuracy
4. Orthogonalization against previous eigenvectors can push into wrong subspaces

**Test Results**:
- Property test with eigenvalues [10, 1, 0.1] fails consistently
- Error: Computed eigenvectors don't satisfy A*v = λ*v
- Symptom: Rayleigh quotients give different eigenvalues than QR algorithm
- Sometimes works for dominant eigenvalue but fails for smaller ones

### Key Files and Functions

**Main Entry Point**:
- `Nx.LinAlg.eig/2` in `lib/nx/lin_alg.ex` (line ~1477)
- Calls `Nx.LinAlg.Eig.eig/2` as fallback implementation

**Implementation** (`lib/nx/lin_alg/eig.ex`):
- `eig/2` (lines 21-30): Handles vectorization/batching
- `eig_matrix/2` (lines 46-108): Main algorithm pipeline
- `balance/2` (lines 219-282): Diagonal scaling for numerical stability
- `hessenberg/2` (lines 284-304): Householder reduction to Hessenberg form
- `qr_algorithm/2` (lines 307-333): Shifted QR iterations → Schur form
- `compute_eigenvectors/4` (lines 415-468): **PROBLEM AREA** - inverse iteration
- `polish_eigenvectors_with_iters/5` (lines 488-526): Refinement via inverse iteration
- `compute_rayleigh_quotients/3` (lines 112-133): Recompute eigenvalues from vectors

**Test**:
- `test/nx/lin_alg_test.exs` (lines 824-877): Property test that constructs A = Q*Λ*Q^H
- Tests: `A * V = V * Λ` (eigenvalue equation)

### Debug History Summary

1. **Initial bug**: Zero eigenvalues due to balance function reshape error → FIXED (commit c8b9f1ac)
2. **Eigenvalue/eigenvector mismatch**: Inverse iteration converged to wrong eigenspaces
3. **Attempted fixes**:
   - Disabled polishing → Still failed
   - Reduced regularization → Marginal improvement
   - Increased iterations → No significant improvement
   - Used Rayleigh quotients → Revealed the mismatch but didn't fix it
   - Used Schur form instead of initial Hessenberg → Better but not sufficient
   - Matching eigenpairs to closest eigenvalues → Greedy matching failed

4. **Current state**: Using Schur form + 60 total iterations of polishing (10 + 50)
   - Success rate: 0/10 on random property test runs
   - Works sometimes for dominant eigenvalue, inconsistent for others

---

## The LAPACK Solution: Backward Substitution on Schur Form

### Overview

LAPACK's `DGEEV`/`ZGEEV` routines use a fundamentally different approach:
**Direct back-substitution on the upper quasi-triangular Schur form** instead of inverse iteration.

### Algorithm: TREVC (Triangular Eigenvector Computation)

After obtaining the Schur form S from QR algorithm:

```
For each eigenvalue λ_i (in reverse order, from smallest to largest):
    1. Set up linear system: (S - λ_i*I) * v_i = 0
    2. Since S is upper quasi-triangular, solve by back-substitution
    3. Normalize v_i
    4. Orthogonalize against previously computed eigenvectors (if needed)
    5. Transform: v_i ← Q * v_i (where Q is from Hessenberg reduction)
```

**Key advantages**:
- More numerically stable than inverse iteration
- Directly uses the structure of the Schur form
- Handles complex conjugate pairs naturally (from 2×2 blocks)
- Well-tested in production code

### LAPACK References

**Primary Routines**:
1. **`DTREVC`** / **`ZTREVC`**: Computes eigenvectors of upper quasi-triangular matrix
   - Source: https://netlib.org/lapack/explore-html/d8/dff/dtrevc_8f.html
   - Complex version: https://netlib.org/lapack/explore-html/d1/d96/ztrevc_8f.html

2. **`DGEEV`** / **`ZGEEV`**: Complete eigenvalue decomposition driver
   - Source: https://netlib.org/lapack/explore-html/d9/d8e/group__double_g_eeigen_ga66e19253344358f5dee1e60502b9e96f.html
   - Shows how TREVC is called in context

**Documentation**:
- LAPACK Users' Guide: https://netlib.org/lapack/lug/
- Section 2.4.8: "Eigenvalue and Singular Value Problems"
- Anderson et al., "LAPACK Users' Guide", 3rd Edition (1999)

**Algorithm Papers**:
- Golub & Van Loan, "Matrix Computations", 4th Edition (2013)
  - Chapter 7.5: "The Practical QR Algorithm"
  - Chapter 7.6: "Invariant Subspace Computation"
- Wilkinson, "The Algebraic Eigenvalue Problem" (1965) - Classical reference

### Reference Implementations

**NumPy/SciPy**:
- Uses LAPACK's `DGEEV`/`ZGEEV` directly via `numpy.linalg.eig`
- Source: https://github.com/numpy/numpy/blob/main/numpy/linalg/linalg.py

**Eigen (C++)**:
- `EigenSolver` class for real matrices
- `ComplexEigenSolver` for complex matrices
- Source: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Eigenvalues/EigenSolver.h
- Implements TREVC-style back-substitution

**Julia**:
- Calls LAPACK directly in `LinearAlgebra.eigen`
- Source: https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/eigen.jl

---

## Implementation Plan for Nx

### Phase 1: Understand TREVC Algorithm (Study)

**Goal**: Fully understand the back-substitution approach

**Tasks**:
1. Read DTREVC source code carefully:
   - How it handles 1×1 blocks (real eigenvalues)
   - How it handles 2×2 blocks (complex conjugate pairs)
   - Scaling strategy to prevent overflow/underflow

2. Study the linear system structure:
   ```
   (S - λ_i*I) * v_i = 0

   For upper triangular S, this becomes:
   For j = n down to 1:
       v_i[j] = -sum(S[j,k] * v_i[k] for k > j) / (S[j,j] - λ_i)
   ```

3. Understand edge cases:
   - Near-zero denominators (S[j,j] ≈ λ_i)
   - Scaling to prevent overflow
   - Complex conjugate pair handling

4. Document the algorithm in pseudocode for Nx `defn`

### Phase 2: Implement Core TREVC Function

**File**: `lib/nx/lin_alg/eig.ex`

**New Function**: `compute_eigenvectors_trevc/4`

```elixir
defnp compute_eigenvectors_trevc(schur, q, eigenvals, opts) do
  # Input:
  #   schur: Upper quasi-triangular Schur form (n×n)
  #   q: Orthogonal matrix from Hessenberg reduction (n×n)
  #   eigenvals: Eigenvalues from Schur form (n)
  #   opts: Options (eps, etc.)
  #
  # Output:
  #   eigenvecs: Eigenvectors of original matrix (n×n, column-wise)

  # Algorithm:
  # 1. For each eigenvalue λ_i (process in reverse order):
  #    a. Check if real or part of complex conjugate pair
  #    b. Solve (S - λ_i*I) * v = 0 by back-substitution
  #    c. Normalize v
  #    d. Store in eigenvecs matrix
  # 2. Transform: eigenvecs = Q * eigenvecs
  # 3. Return eigenvecs
end
```

**Key Challenges**:
1. **Back-substitution in `defn`**: Need to implement column-by-column using `while` loops
2. **Scaling**: Implement scaling to prevent overflow (similar to TREVC)
3. **Complex pairs**: Handle 2×2 blocks on diagonal of Schur form
4. **Numerics**: Small denominators need careful handling

**Implementation Strategy**:
```elixir
# Pseudocode structure:
{eigenvecs, _} =
  while {eigenvecs, {i = n-1}}, i >= 0 do
    lambda = eigenvals[i]

    # Initialize eigenvector (start with v[i] = 1)
    v = initialize_eigenvector(i, n)

    # Back-substitution from bottom to top
    {v, _} =
      while {v, {j = i-1}}, j >= 0 do
        # Compute: v[j] = -sum(S[j,k] * v[k] for k > j) / (S[j,j] - lambda)
        sum = compute_sum(schur, v, j, i)
        denom = schur[j,j] - lambda
        denom_safe = max(abs(denom), eps)
        v = put_v_entry(v, j, -sum / denom_safe)
        {v, {j - 1}}
      end

    # Normalize
    v = v / norm(v)

    # Store in eigenvecs
    eigenvecs = put_column(eigenvecs, i, v)

    {eigenvecs, {i - 1}}
  end
```

### Phase 3: Replace Inverse Iteration

**Modify** `eig_matrix/2` in `lib/nx/lin_alg/eig.ex`:

```elixir
# BEFORE (lines ~85-90):
eigenvecs = compute_eigenvectors(schur, q, eigenvals, opts)
eigenvecs = polish_eigenvectors_with_iters(a, eigenvals, eigenvecs, opts, 10)

# AFTER:
eigenvecs = compute_eigenvectors_trevc(schur, q, eigenvals, opts)
# No polishing needed - TREVC is already accurate!
```

**Remove/deprecate**:
- `compute_eigenvectors/4` (old inverse iteration version)
- Most polishing steps (may keep light polish for very close eigenvalues)
- `match_eigenpairs/4` (no longer needed)

### Phase 4: Testing & Refinement

**Test Suite**:
1. Run existing tests (diagonal, triangular, rotation, batched)
2. Run property test with well-separated eigenvalues
3. Run property test with close eigenvalues [10, 1, 0.1]
4. Edge cases:
   - Repeated eigenvalues
   - Zero eigenvalues
   - Very large/small eigenvalues (conditioning)
   - Complex eigenvalues (rotation matrices)

**Success Criteria**:
- Property test passes consistently (>95% success rate over 100 runs)
- Accuracy: `||A*V - V*Λ|| / ||A|| < 10^-4` for f32
- Performance: Similar or better than current implementation

### Phase 5: Optimization & Documentation

**Optimizations**:
1. Vectorize operations where possible within `defn` constraints
2. Reduce memory allocations
3. Profile and optimize hotspots

**Documentation**:
1. Update module documentation with algorithm description
2. Add inline comments explaining TREVC approach
3. Reference LAPACK and papers
4. Document numerical properties and limitations

---

## Alternative Approaches (If TREVC Doesn't Work)

### Option A: LAPACK FFI Binding
**Pros**: Proven, highly optimized
**Cons**: External dependency, platform-specific compilation

### Option B: Jacobi Algorithm
**Pros**: Simultaneously computes eigenvalues and eigenvectors, naturally parallel
**Cons**: O(n³) per sweep, may need many sweeps, only for symmetric matrices

### Option C: Arnoldi/Lanczos Iteration
**Pros**: Good for finding a few eigenvectors, iterative refinement
**Cons**: Complex to implement in `defn`, better for sparse matrices

### Option D: Accept Current Limitations
**Pros**: Already implemented
**Cons**: Unreliable for close eigenvalues, user-facing failures

---

## Estimated Effort

**Phase 1** (Study): 2-4 hours
- Read DTREVC source code
- Understand algorithm details
- Write pseudocode

**Phase 2** (Implementation): 8-12 hours
- Implement `compute_eigenvectors_trevc/4`
- Handle edge cases (scaling, small denominators)
- Debug initial version

**Phase 3** (Integration): 2-4 hours
- Replace old eigenvector computation
- Clean up unused code
- Update call sites

**Phase 4** (Testing): 4-6 hours
- Fix bugs found in testing
- Handle edge cases
- Achieve acceptable accuracy

**Phase 5** (Polish): 2-4 hours
- Documentation
- Code review feedback
- Performance optimization

**Total**: 18-30 hours (depending on complexity of edge cases)

---

## References Summary

**Essential Reading**:
1. LAPACK DTREVC: https://netlib.org/lapack/explore-html/d8/dff/dtrevc_8f.html
2. Golub & Van Loan, "Matrix Computations" (Chapter 7)
3. Current Nx implementation: `lib/nx/lin_alg/eig.ex`

**For Complex Eigenvalues**:
1. LAPACK ZTREVC: https://netlib.org/lapack/explore-html/d1/d96/ztrevc_8f.html
2. Handling 2×2 blocks in real Schur form

**Testing Reference**:
- NumPy's `numpy.linalg.eig` for validation
- Test matrices from `test/nx/lin_alg_test.exs`

---

## Contact & Questions

For questions about this implementation:
- Review LAPACK documentation first
- Check Golub & Van Loan for theoretical background
- Look at NumPy/Eigen source for practical examples
- Test incrementally with simple cases (diagonal, 2×2 matrices)

The key insight is: **Use the structure of the Schur form directly via back-substitution** rather than fighting with inverse iteration.
