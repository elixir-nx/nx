# Continue Evaluator Cache Flattening - Fix Remaining Test Failures

## Context
I'm working on refactoring `nx/lib/nx/defn/evaluator.ex` to use a flattened cache format. The implementation is 88% complete (43/49 tests passing). All code has been committed to branch `feature/evaluator-cache-flattening`.

## Current State

### ✅ Working (43/49 tests)
The flattened cache format works perfectly for:
- Basic arithmetic/tensor operations
- Simple cond/if/while
- Functions, tuples, most hooks
- Parent scope references in simple cases

### Cache Format Implemented
```elixir
# Generic ops now store:
id => {:expr, count, type, shape, names, vectorized_axes, op, args}

# Scoped ops still use integer format:
id => integer_count  # For :elem, :attach_token, :cond, :while, :fun, :token, :optional
```

### Key Implementation Details
- **compute_cache_op** (line ~370): Creates flattened entries for generic ops
- **eval** (line ~416): Handles flattened entries (with/without cached results)
- **eval_parent** (line ~462): Looks up flattened entries in parent cache stack
- **decrement_parents** (line ~497): Decrements parent reference counts
- Parent refs stored as full tensors temporarily, to be replaced with flattened entries

## 🐛 Problem: 6 Test Failures (ALL involve nested scopes)

Run `cd nx && mix test test/nx/defn/evaluator_test.exs --seed 0` to see failures:

1. **test decompositions lu** (line 61) - While loop with nested cond
2. **test decompositions svd** (line 89) - While loop with nested cond
3. **test cond cache on both** (line 555) - Hook + attach_token + nested cond
4. **test cond cache with nested map** (line 580) - Nested cond (2-3 levels)
5. **test cond cache with nested condition** (line 622) - Triple-nested cond
6. **test vectorization vectorize works inside defn** (line 640) - Vectorized ops

### Error Pattern
All fail with **"trying to read evaluator cache that has expired"** - entries are deleted too early.

### Root Cause
When an expression is used across **multiple nesting levels** (e.g., outer scope + inner cond + nested cond), the reference count is too low:
- Expression `E` defined in outer scope
- Used in 2+ different nested scopes (e.g., two cond branches + outer)
- Count should be 3+, but is only 2
- Gets deleted after 2 uses, but 3rd use fails

## 🎯 Your Task
Fix the reference counting for nested scopes so all 49 tests pass.

## 💡 Investigation Approach

1. **Add debug output** to understand reference counting:
   - Print counts when creating flattened entries
   - Log when entries are deleted
   - Trace parent_ids through nested cond building

2. **Focus on simplest failure first**: "cond cache with nested map" (line 580)
   - Simpler than decompositions (no while loops)
   - Just nested conds with shared parent expressions

3. **Check compute_cache_op(:cond)** (line ~287):
   - How it processes `clause_caches`
   - How it merges parent entries (lines 302-338)
   - The `seen_ids` logic might not count correctly

4. **Verify decrement_parents** (line ~497):
   - Gets called at line 540 after cond evaluation
   - Should decrement each parent_id once
   - But if parent was also used outside cond, might be deleted incorrectly

## 📝 Key Files
- `/Users/valente/.cursor/worktrees/nx/wex/nx/lib/nx/defn/evaluator.ex` - Main file
- `/Users/valente/.cursor/worktrees/nx/wex/nx/test/nx/defn/evaluator_test.exs` - Tests
- `/Users/valente/.cursor/worktrees/nx/wex/EVALUATOR_FLATTENING_STATUS.md` - Detailed status
- `/Users/valente/.cursor/worktrees/nx/wex/REFACTORING_SUMMARY.md` - Implementation summary

## 🔍 Specific Things to Check

1. In `compute_cache_op(:cond)` around line 302-338:
   - When flattened entries from branches are merged into parent cache
   - Check if Map.put_new correctly handles cases where entry already exists
   - Verify count increments work for entries seen in multiple branches

2. In `compute_cache` around line 214-233:
   - When incrementing flattened entry counts
   - Full tensor replacement logic (line 219-233)
   - Might be resetting count to 1 when should increment

3. Reference counting during cache building vs evaluation:
   - Build-time counts: how many times expression appears in tree
   - Run-time decrements: how many times expression is evaluated
   - These must match exactly

## ✅ Success Criteria
- All 49 tests in `test/nx/defn/evaluator_test.exs` pass
- Commit fixes with clear messages
- Run full Nx test suite to ensure no regressions
- Update status documents with resolution

## 🚫 Don't Do
- Don't disable flattening or revert to old format
- Don't skip failing tests
- Don't change test expectations
- The architecture is correct, just needs reference counting fix

## 🔧 Debugging Commands

```bash
# Run specific failing test with seed
cd /Users/valente/.cursor/worktrees/nx/wex/nx
mix test --only line:580 --seed 0 test/nx/defn/evaluator_test.exs

# Run all evaluator tests
mix test test/nx/defn/evaluator_test.exs --seed 0

# Add debug output by setting env var and adding IO.puts in eval_parent
DEBUG_CACHE=1 mix test --only line:580 test/nx/defn/evaluator_test.exs
```

Good luck! The hard part is done - just need to fix the counting logic for nested scopes.
