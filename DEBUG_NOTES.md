# Debug Notes for Remaining 5 Test Failures

## Current Status: 44/49 passing (90%)

## Key Findings

### Bug Location Identified
The 5 failing tests all involve integer-format operations (`:attach_token`, `:elem`, `:cond`) that are NOT found in the cache during evaluation.

**Error Pattern**: "trying to read evaluator cache that has expired"  
**Op Types**: :attach_token (3 tests), :cond (2 tests)

### Root Cause Hypothesis

**Integer operations in cond branches stay in branch cache** (line 337-338):
```elixir
{id, counter}, seen_ids_cache ->
  {[{id, counter}], seen_ids_cache}  # Kept in clause_cache, not merged to parent
```

**But cond evaluation discards branch cache** (line 609):
```elixir
{res, [_ | caches]} = composite_eval(chosen, state, chosen_cache)  # Discards first cache
```

**Question**: How does the original code make this work?
- Original code has SAME structure (verified by checking b760c7a1)
- Integer counters stay in clause cache in original too
- Yet original passes all tests

### What I've Tried

1. ✅ **No-deletion policy**: Keep flattened entries even with count <= 0
   - Status: Implemented, helps but doesn't solve core issue
   
2. ✅ **Map.put instead of Map.put_new**: Replace parent refs with flattened entries
   - Status: Implemented, correct fix for a different issue
   
3. ✅ **Special handling for inline ops**: parameter/constant/tensor/metadata in eval_parent
   - Status: Implemented, fixed 1 test (nested map)
   
4. ✅ **Direct evaluation of parent ref tensors**: eval(stored_tensor) in eval_parent
   - Status: Implemented, correct approach
   
5. ❌ **Merging integer counters to parent**: Tried merging integer ops to parent cache
   - Status: Reverted, broke 42 additional tests
   
6. ❌ **Not decrementing integers in decrement_parents**: Keep integer entries unchanged
   - Status: Tried, didn't help

### Key Insight

The failing tests involve:
- **Multiple conds** using the same hooked expression (`res` used in `left` and `right` conds)
- **attach_token** operations created implicitly when hooked expression is used
- **Cross-cond references**: attach_token from one cond might reference expr from another

### Missing Piece

The attach_token operation is literally NOT in the cache at all - not in branch cache, not in parent cache, nowhere. This suggests:

**Possibility 1**: attach_token isn't being added during cache building
- But compute_cache_op for attach_token IS defined (line 366-369)
- It processes args and should keep the `id => 1` entry
- **BUG**: Maybe the id => 1 entry is being overwritten/deleted during args processing?

**Possibility 2**: attach_token IS added, but in wrong scope
- Maybe it's added to a different cache level
- Maybe parent_ids incorrectly includes attach_token's ID
- **BUG**: Scope tracking error?

**Possibility 3**: Multiple attach_tokens with same ID
- If `res` is used multiple times, multiple attach_tokens created
- Maybe they share an ID due to CSE?
- **BUG**: ID collision?

## Next Debugging Steps

###1. Add debug output during cache building
```elixir
# In compute_cache at line 205, log when op == :attach_token:
IO.puts("compute_cache :attach_token #{inspect(id)}")
IO.puts("  parent_ids: #{Map.has_key?(state.parent_ids, id)}")
IO.puts("  cache before: #{Map.has_key?(cache, id)}")
# After compute_cache_op, check if id is in result
```

### 2. Trace attach_token through full lifecycle
- Cache building: Is it added? What value?
- Cache merging (cond): Is it kept in clause cache?
- Evaluation: Is it in chosen_cache?
- Failure: Which cache level was it supposed to be in?

### 3. Compare with original code execution
- Instrument original code with same debug output
- Run same test
- Compare cache contents at each step
- Find where our implementation diverges

### 4. Check if attach_token is a parent ID
During cond cache building, check:
```elixir
Tree.scope_ids(clause, current_ids)
```
Does this include attach_token's ID? If yes, it will be stored as parent ref.

## Specific Test Analysis

### Test: cond cache on both (line 555)
```elixir
res = hook(a + b, :example, ...)  # Creates token wrapping (a+b)

left = if bool do res else -res end   # Uses res -> creates attach_token
right = if bool do res * 2 else res end  # Uses res -> creates MORE attach_tokens

left * right
```

**Structure**:
- 2 conds (left, right)
- Each cond has 2 branches (true, false)  
- Each branch uses `res` -> creates attach_token
- Total: 4 attach_token operations

**Question**: Which attach_token is missing from cache?
- Is it all 4?
- Just one specific one?
- The one being evaluated when error occurs?

## Code Locations

- compute_cache for attach_token: Line 366-369
- compute_cache main entry: Line 205-236
- cond cache building: Line 284-345
- eval for integer ops: Line 444-448
- eval_parent for integer ops: Line 499-502

## Success Criteria

- [ ] All 49 evaluator tests pass
- [ ] Understand why original code works
- [ ] Fix without breaking existing 44 tests
- [ ] Verify full test suite (no regressions beyond the 5)
