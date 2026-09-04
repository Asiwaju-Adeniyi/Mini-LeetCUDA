# CUTE Layout Representation and Algebra — Study Notes

Working through Cecka's CUTE preprint and Colfax's categorical-foundations
material, page by page, with my own worked examples.

## Progress
| Section | Status | Last updated |
|---|---|---|
| 1.2 Canonical loops | done | 2026-08-24 |
| 1.3 Tensors and folding | done | — |
| 2.1 Tuples and HTuples | done | — |
...

## 1. Introduction and motivation
Xintong (MTS, Thinking Machines) pointed me toward CUTLASS/CuTe — studying
it on the side while writing FlashAttention forward/backward.

### 1.2 Canonical loops and loop transformations
**Core idea:** a canonical loop nest is fully characterized by Shape:Stride.
Shape defines the domain (coordinates); stride defines the codomain (memory
offsets); layout = stride∘shape.

`for(m=2; m<=16; m+=3)` relates to a canonical `for(i=0;...)` via
`m = start + step·i = 2 + 3i`: `i=0→m=2`, `i=1→m=5`, `i=2→m=8`.

**Why it matters:** a transformation of a Shape:Stride is itself another
Shape:Stride object — transformations and the things they transform live
in the same representation.

### 1.3 Tensors and folding
Modes classify by which operands they appear in:
- row: A, C (not B) · column: B, C (not A)
- reduction: A, B (not C) · batch: A, B, C

8-element array `a`–`h`, viewed as `(2,2,2):(2,1,4)` — row-step 2, col-step
1, batch-step 4.

**Fold mode 2 → mode 0:** flat `(4,2):(2,1)`, CUTE `((2,2),2):((2,4),1)`.
Works because `row-step × row-size = batch-step` (`2×2=4`) — the two
original steps chain cleanly into one.

**Fold mode 2 → mode 1:** glued offsets jump `+1,+3,+1` — not constant, so
no flat stride exists (`✗`). CUTE still works with zero extra computation:
`(2,(2,2)):(2,(1,4))` — the original steps, untouched, side by side.

## 2. Layout representation
### 2.1 Tuples and HTuples

**Tuple(T):** ordered list, all entries from the same set `T`. `rank(X)` =
slot count; `X_i` = entry at slot `i`.
Ex: `(4,2)` is `Tuple(Z⁺)`, `rank=2`, `X_0=4`, `X_1=2`.

**HTuple(T):** a bare element of `T`, or a Tuple of HTuple(T)s — recursive.
- `rank`: top-level slots only. Bare element → 1 (by definition). Nesting
  inside a slot doesn't change rank: `rank(((2,2),2)) = 2`.
- `depth`: bare element → **0** (base case). Tuple → `1 + max(depth of
  entries)` — set by the deepest branch, not the average.
  - `depth((2,(4,1),-1)) = 2`: leaves are 0, `(4,1)` is `1+max(0,0)=1`,
    whole thing is `1+max(0,1,0)=2`.
  - `depth(((4,6),(3,(2,2),8))) = 3`: deepest chain is root→`(3,(2,2),8)`→
    `(2,2)`→leaf, three tuple-in-tuple levels.
  - `depth(((2,2),2)) = 2`: matches my own fold-1 CUTE shape.

**Congruence (∼):** same nesting *shape*, values irrelevant. Equivalence
relation (symmetric). Slot check: leaf/leaf ✓; tuple/tuple same rank →
recurse; any leaf/tuple mismatch, or tuple/tuple different rank → ✗. One
bad slot anywhere fails the whole thing.

**Weak congruence (≲), "P coarsens S":** partial order (not symmetric). A
leaf on **P's side only** is a free pass regardless of what S has there;
tuple on P's side demands a matching-rank tuple on S's side, then recurse.
`(m,4) ≲ ((a,b),4)` holds; the reverse doesn't.

**Why congruence/weak-congruence matter:** weak congruence is exactly what
lets a shape accept both full ND coordinates and a coarser 1D coordinate at
once (2.2) — same idea as splitting a thread index into warp/lane by hand.