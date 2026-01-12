# Evaluator Cache Flattening - Status Report

## Overview
This document tracks the progress of refactoring the Nx.Defn.Evaluator to use a flattened cache structure, eliminating exponential growth from duplicated subexpressions.

## Problem Statement
Code like `x = a + b; x * x` expands to `(a + b) * (a + b)` in the expression tree, causing exponential growth. When the compiled function is transferred (e.g., for distributed execution), the enclosed expression can be extremely large.

## Solution Approach
Transform the evaluator to use a **flattened graph representation**:
- Store expressions as `{:expr, count, type, shape, names, vectorized_axes, op, args}` instead of full tensor structs
- Args contain IDs or raw values, not full expression trees
- Cache becomes `id => flattened_entry` instead of `id => full_tensor`

## Completed Work

### ✅ Phase 1: Helper Functions
- Added `arg_to_id_or_value/1` and `args_to_ids_or_values/1` for ID extraction
- Added `reconstruct_tensor/5` for rebuilding tensors from cached metadata
- Added `make_tensor/5` for creating tensor wrappers

### ✅ Phase 3: Leaf Node Refactoring
- Reorganized `compute_cache` to separate leaf nodes (`:constant`, `:tensor`, `:parameter`)
- These nodes don't store cache entries (evaluated inline)
- Maintained backward compatibility with existing tests

### ✅ Phase 4: Generic Operation Flattening
- Implemented flattened cache format in `compute_cache_op/4`
- Generic operations now store `{:expr, 1, type, shape, names, vectorized_axes, op, args}`
- Updated `compute_cache` to increment counts for flattened entries
- Excluded `:elem` and `:attach_token` from flattening (they have special eval logic)

### ✅ Eval Logic Updates
- Updated `eval/3` to handle flattened entries (with and without cached results)
- Updated `eval_parent/6` to look up flattened entries in parent scopes
- Updated `decrement_parents/2` to decrement flattened entry counts
- Maintained backward compatibility with legacy integer-count format

### ✅ Cond Cache Updates
- Updated `:cond` cache building to handle flattened entries from branches
- Flattened entries for parent expressions are stored in parent cache
- Both flattened and full-tensor formats are handled

## Current Status

### ✅ Working (43/49 tests - 88%)
- Simple arithmetic operations (add, multiply, etc.)
- Basic tensor operations (reshape, iota, etc.)
- Basic cond/if operations
- Simple while loops
- Tuple operations
- Basic hooks
- Most vectorization

### ❌ Failing (6/49 tests - 12%)
1. **decompositions lu** - Complex while loop with nested cond
2. **decompositions svd** - Complex while loop with nested cond
3. **cond cache on both** - Hook/token with attach_token + parent refs
4. **cond cache with nested map** - Nested cond with map containers
5. **cond cache with nested condition** - Triple-nested cond
6. **vectorization vectorize works inside defn** - Vectorized_axes handling

## Core Issue: Parent Scope References in Cond

### The Problem
When a cond branch references a parent expression:
```elixir
c = as_type b              # Stored as flattened entry in main cache
e = cond a -> c, true -> d # Cond branch references c
```

The evaluation fails with "cache expired" error because:
1. `c` is stored as a flattened entry `{:expr, ...}` in the main cache
2. When building the cond cache, `c` is identified as a parent expression
3. The flattened entry is copied to the parent cache
4. When evaluating the branch, the lookup for `c` fails

### Why It Fails
The exact failure point is in `eval_parent/6` - when looking up a parent expression, it searches through the cache stack but doesn't find the entry. Possible causes:
1. The flattened entry isn't being stored in the right cache level
2. The cache stack isn't being constructed correctly for cond evaluation
3. The flattened entry is being deleted/modified before the lookup
4. The ID matching logic doesn't work correctly for flattened entries

### Investigation Needed
1. Add debug output to trace cache contents during cond evaluation
2. Verify that parent flattened entries are actually in the cache stack
3. Check if the issue is with how `cond_clause/5` constructs the cache stack
4. Verify that `composite_eval` is called with the correct cache stack

## Remaining Work

### High Priority
1. **Fix cond parent scope lookups**
   - Debug why flattened parent entries aren't found during branch evaluation
   - Ensure cache stack is correctly constructed for cond branches
   - May need to adjust how parent entries are stored/retrieved

2. **Fix while loops**
   - Similar parent scope issues as cond
   - While body needs access to parent cache entries

3. **Fix fun operations**
   - Anonymous functions with closures over parent expressions
   - Similar parent scope reference issues

### Medium Priority
4. **Handle hooks correctly**
   - Token operations with hooks
   - Ensure hook expressions can access parent cache

5. **Fix tuple operations**
   - `:elem` operations in cond branches
   - Tuple destructuring with parent references

### Low Priority
6. **Refactor precompile to return root IDs**
   - Currently returns full expression tree
   - Should return just IDs with cache containing all metadata
   - Requires all operations to work with flattened format first

7. **Optimize memory usage**
   - Remove unused helper functions (arg_to_id_or_value, etc.)
   - Consider compacting cache entries further
   - Profile memory usage before/after

## Technical Debt
- Legacy integer-count format still supported for backward compatibility
- Mixed flattened/non-flattened formats during transition
- Some operations (`:elem`, `:attach_token`) excluded from flattening
- Scoped operations (`:cond`, `:while`, `:fun`) have complex cache management

## Next Steps
1. Add comprehensive debug logging to cond evaluation
2. Create minimal failing test case for parent scope lookup
3. Fix the core parent scope issue
4. Incrementally enable flattening for more operation types
5. Remove legacy format support once all tests pass
6. Implement Phase 5 (precompile root IDs)

## Performance Considerations
- Flattened format should reduce memory usage for large expressions
- Lookup overhead might increase slightly (tuple matching vs integer)
- Reference counting still works the same way
- Cache size should be significantly smaller for expressions with sharing

## Testing Strategy
- Run full test suite after each change
- Focus on cond/while/fun tests first (they're the blockers)
- Add new tests for edge cases (deep nesting, multiple parent refs, etc.)
- Verify memory usage improvement once working

## Conclusion
The flattened cache infrastructure is in place and working for simple operations. The main blocker is parent scope references in scoped operations (cond/while/fun). Once this is fixed, the remaining work is incremental refinement and optimization.

The core architectural change (flattening) is sound, but the implementation needs careful handling of scope boundaries and parent references.

