# CUTE Layout Representation and Algebra — Study Notes

Working through Cecka's CUTE preprint and Colfax's categorical-foundations
material, page by page, with my own worked examples.

## Progress
| Section | Status | Last updated |
|---|---|---|
| 1.2 Canonical loops | done | 2026-08-24 |
| 1.3 Tensors and folding | done |  |
| 2.1 Tuples and HTuples | done |  |
| 2.2 Shape | done |  |
| 2.2.1 Coordinate Sets/Compatibility | Ongoing |  |
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

### 2.2 Shape

A shape is officially just an `HTuple(Z⁺)` — the formal name for what I'd
already been building with folding. Its size `|S|` is the product of its
elements (recursive if nested): `|((2,2),2)| = (2×2)×2 = 8`.

Big idea: the *same* data can be addressed at different "zoom levels" —
fully split apart, partially glued, or fully flat — and all of these are
legal because each coarser shape is weakly congruent to the refined one.
This is exactly *why* a generic algorithm (GEMM, COPY) can accept any
oddly-folded tensor: the real shape just has to coarsen into the shape the
algorithm expects.

### 2.2.1 Coordinate Sets (so far)

The coordinate set of a shape `S`, written `Z_S`, is built by the same
recipe as the shape itself — swap each number `N` for its range `{0,...,
N-1}`, and nesting becomes Cartesian product. This is called `S`'s
**natural coordinates**.

Example: shape `(3,4)` → `Z_3 × Z_4`, listed fastest-first (colex order):
`(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),...`

Note: a mode of size 1 gives `Z_1 = {0}` — a dimension that always reads 0,
still counts toward rank, but adds no extra coordinates.

Other coordinate sets are valid for the same data too — any shape that
*coarsens* `S` has its own natural coordinates. This is just naming what I
already saw with box `f`: `(i0,i1,i2)`, `(r,i1)`, and `k` were three
different shapes' natural coordinates, all describing the same box.

### 2.2.1 Compatibility (Def 2.7)

Compatibility (`⪯`) looks like weak congruence's twin — same coarsen/refine
language, same partial-order shape — but it checks something different:
**size**, not nesting structure.

Base case is where the two relations actually diverge:
| | Weak congruence (≲) | Compatibility (⪯) |
|---|---|---|
| P is a leaf | auto-pass, S can be anything | P must **equal `\|S\|`** — real arithmetic |
| P, S both tuples | same rank, recurse per slot | same rank, recurse per slot |

Tuple case is identical in both. The whole difference is the leaf case:
weak congruence's leaf is a structural wildcard (no numbers checked);
compatibility's leaf demands an actual size match.

**Worked examples, from the text:**
- `30 ⪯ (2,15)`: `30 = |(2,15)| = 30` ✓ (leaf vs. total size).
- `(2,15) ⪯ (2,(3,5))`: slot 0: `2=|2|` ✓; slot 1: `15=|(3,5)|=15` ✓.
- `(2,(3,5))` and `((3,2),5)`, same size (30) both ways, but NOT compatible:
  checking one direction, slot 0 needs `2=|(3,2)|=6` — fails.

**Compatibility vs. weak congruence — genuinely independent, proven both ways:**
- `30 ⪯ ((3,2),5)` (sizes match: `30=6×5`) but `((3,2),5) ≴ 30` — top level
  is tuple-vs-leaf, a structural mismatch weak congruence can't get past
  (compatible, not weakly congruent).
- `(7,10) ≲ ((2,3),5)` (both of `(7,10)`'s slots are leaves → auto-pass,
  structure never even inspects `S`) but `(7,10) ⪯̸ ((2,3),5)` — slot 0
  needs `7=|(2,3)|=6`, fails (weakly congruent, not compatible).

Neither relation implies the other. One checks nesting shape, the other
checks element counts.

**Ties to my own fold examples:** `(4,2) ⪯ ((2,2),2)` and `(2,4) ⪯
(2,(2,2))` — **both** compatible, since folding never changes total size.
Compatibility says nothing about whether a flat stride exists — that's a
*stride* fact (Fig. 1's `✗`), not a *shape* fact.