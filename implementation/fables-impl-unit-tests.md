# Fable's Implementation: Unit Tests

There are two test files — one for the compression math (standalone, runs without GPU), one for the scheduler accounting.

---

File 1: tests/v1/worker/test_kv_compression_standalone.py

Setup: loading without vLLM (lines 27-80)

The module loader (_load_module) stubs out vLLM-internal imports (logger, cache interface) so the compression math can be tested on a plain CPU machine. This means you can run it with just python tests/v1/worker/test_kv_compression_standalone.py — no GPU, no full vLLM install needed.

Constants: BLOCK_SIZE=16, NUM_HEADS=4, HEAD_SIZE=32, CPU, float32.

Two helpers:
- make_paged_cache(num_blocks, num_layers) — creates the [2, num_blocks, block_size, num_kv_heads, head_size] tensors
- write_sequence(kv_caches, block_row, keys, values) — writes dense K/V into the paged cache using _slots_for_positions + scatter_slots

---

Test 1: test_gather_scatter_roundtrip (line 110)

What it checks: write K/V to the paged cache with scatter_slots, read them back with gather_slots, verify you get the exact same tensors.

Why it matters: if gather/scatter don't roundtrip correctly, nothing else works. This is the foundation — physical address translation must be lossless.

Uses a non-trivial block row [5, 2, 7, 0] (non-contiguous physical blocks) with 37 tokens (not a multiple of block_size — exercises the partial-block case).

---

Test 2: test_keydiff_scores_match_kvpress (line 122)

What it checks: our keydiff_scores produces the same output as kvpress's KeyDiffPress.score().

Why it matters: proves that Fable's extraction of the KeyDiff scoring math from kvpress is numerically correct. The permutation keys.permute(1, 0, 2).unsqueeze(0) converts from our layout [T, H, D] to kvpress layout [batch, H, T, D].

Skips gracefully if kvpress is not installed.

---

Test 3: test_compact_request_kv (line 148)

What it checks: full replacement compaction on 3 layers, 100 tokens, ratio 0.5. After compaction:
1. n_kept == 50 (the target)
2. For each layer and head: the kept keys/values in the cache match the per-head top-50 by KeyDiff score, in temporal order

How it verifies: independently computes scores.topk(n_kept).indices.sort() per head, then checks that gather_slots returns those exact rows. This is the full replacement pipeline tested end-to-end — score, select, scatter — verified against a reference that doesn't use the compression code at all.

---

Test 4: test_compact_noop_when_ratio_keeps_all (line 184)

What it checks: edge case — a 1-token sequence with n_target=0. The clamp to max(1, n_target) means n_kept=1, and the cache is unchanged.

Why it matters: guards against the boundary where compression ratio would try to discard everything.

---

Test 5: test_masked_keydiff_scores_match_per_head (line 199)

What it checks: masked_keydiff_scores with ragged per-head valid lengths produces the same scores as running keydiff_scores independently on each head's valid subset. Invalid positions must be -inf.

Why it matters: this is the scoring function used by filtering. Heads have different valid lengths ([30, 17, 5, 23]), so the mask must correctly isolate each head's valid tokens. The anchor (mean of L2-normalized keys) must be computed only over valid entries — if invalid positions leak in, the anchor shifts and scores are wrong.

---

Test 6: test_filtering_step_multistep_vs_kvpress (line 218)

What it checks: this is the big one. Runs 60 decode steps on 2 layers, comparing after every single step:
1. Per-(layer, head) valid lengths match kvpress's FilteringPress._lengths
2. Per-head packed K/V in the paged cache matches kvpress's buffers (using fill_padding=False, which is your PaddedTensor semantics)
3. The shared physical length equals max(buffer_lengths) across layers

How it works:
- vLLM side: writes new token at the shared column, calls filtering_step, tracks num_kv_discarded
- kvpress side: appends to a dense buffer, calls FilteringPress.compress()
- After each step, gathers the paged cache and compares head-by-head against kvpress's dense buffers

Behavioral checks at the end: both keeps and skips occurred, heads diverged (different lengths), and the final shared length is meaningfully below logical length.

This is the strongest test in the suite — it proves that Fable's paged-cache filtering is numerically equivalent to your FilteringPress implementation in kvpress.

---

Test 7: test_filtering_step_keeps_all_when_under_target (line 362)

What it checks: when all heads have fewer valid tokens than n_kept, every head accepts the new token, no column is freed. new_len == num_cached and all lengths increment.

Why it matters: guards the early-decode phase where compression hasn't built up enough tokens to start skipping.

---

Test 8: test_filtering_realized_ratio_converges (line 381)

What it checks: over 1200 decode steps, the realized per-head keep ratio lands near 1 - ratio = 0.5 (within [0.35, 0.65]), and the shared/memory ratio also falls within a reasonable band.

Why it matters: this is your ratio-as-expectation property. On random keys (no semantic bias), the long-run realized compression tracks the target. This is a statistical test — one seed, loose bands, but it validates the self-regulating feedback loop where n_kept = round(logical_total * (1 - ratio)) uses the full logical count.

---

Test 9: test_iterative_compaction (line 441)

What it checks: repeated compaction across growth phases (chunked prefill + decode), verified against an independently maintained per-head reference list.

The sequence: append 30 tokens → compact → append 30 more → compact → 3 rounds of append 8 + compact. At the end: logical_total=84, num_cached = int(84 * 0.5) = 42.

How it verifies: maintains ref_k[h] and ref_v[h] lists per head, independently scores and selects top-k after each compaction, then checks the paged cache matches. This exercises the composite survivor property — after compaction, the kept set is a mix of survivors from multiple rounds, and re-scoring works correctly on that mix.

---

Test 10: test_manager_phase_coverage (line 554)

What it checks: the KVCompressionManager trigger logic for both algorithms through all phases: chunked prefill (3 chunks of 16), prefill completion, and decode.

For full_replacement:
- Each prefill chunk triggers compaction (interval reached)
- Decode: no compaction until interval elapses (step 15), then one batch drop of 8
- kv_filter_lengths remains None

For filtering:
- Same prefill chunks trigger compaction (filtering uses full replacement during prefill)
- Decode: per-head filtering runs every step (delta is 0 or 1)
- kv_filter_lengths is populated
- Both keeps and skips occur

This test validates the three-branch logic in run_post_forward and the kv_compressed handoff flag.

---

File 2: tests/v1/core/test_kv_compression_scheduler.py

This tests the scheduler side — how num_kv_discarded flows from the model runner to the scheduler and affects block allocation.

Test 1: test_kv_compression_discarded_accounting (line 35)

What it checks: end-to-end scheduler flow with full replacement:
1. Prefill 100 tokens → 7 blocks (ceil(100/16))
2. Model runner reports 50 discarded → request.num_kv_discarded = 50, logical accounting unchanged
3. Decode 40 steps → physical occupancy goes from 50 to 90, still fits in 7 blocks (no new allocation!)
4. After 70 more decode steps → physical occupancy crosses 112 → new block allocated, ending at 10 blocks

Key insight: without compression the same logical length (210) would need ceil(210/16) = 14 blocks. With compression: 10 blocks. This proves the scheduler correctly uses physical lengths for block allocation while keeping logical accounting untouched.

Test 2: test_kv_compression_filtering_accounting (line 86)

What it checks: scheduler flow with filtering — every other decode token is skipped (delta=1 on even steps). After 32 decode steps: num_kv_discarded=16, 3 blocks instead of the 4 that would be needed without compression.

Test 3: test_kv_compression_reset_on_preemption (line 116)

What it checks: when a request is preempted, num_kv_discarded resets to 0 along with num_computed_tokens. This ensures that if the request is rescheduled, it starts fresh — no stale compression state.

---

Test architecture summary

┌──────────────────────────┬────────────────────────────────────────────────────┐
│           Test           │                 What it validates                  │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ gather/scatter roundtrip │ Physical address translation                       │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ keydiff vs kvpress       │ Scoring correctness                                │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ compact_request_kv       │ Full replacement pipeline (score → topk → scatter) │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ compact noop             │ Edge case: can't compress below 1 token            │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ masked_keydiff_scores    │ Per-head ragged scoring for filtering              │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ filtering vs kvpress     │ Step-by-step equivalence with your FilteringPress  │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ filtering keeps all      │ Edge case: under-target → accept all               │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ realized ratio converges │ Ratio-as-expectation statistical property          │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ iterative compaction     │ Repeated compaction with composite survivors       │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ manager phase coverage   │ Three-branch trigger logic, both algorithms        │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ scheduler discarded      │ Block allocation uses physical, not logical        │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ scheduler filtering      │ Online filtering accounting                        │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ scheduler preemption     │ Compression state resets on preempt                │
└──────────────────────────┴────────────────────────────────────────────────────┘

That's all four traces complete. The standalone tests run without GPU and validate math equivalence with kvpress. The scheduler tests validate the accounting that connects the model runner to the block allocator.