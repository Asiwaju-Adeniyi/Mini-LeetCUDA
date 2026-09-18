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

### 2.2.1 (cont'd) — Z(S), the set of compatible coordinate sets (Def 2.8)

`Z(S)` collects *every* legal coordinate-naming-scheme for a shape `S` —
one entry for each shape `S'` that coarsens `S` (`S' ⪯ S`), taking that
scheme's own natural coordinates `Z_S'`.

Two entries are always guaranteed:
- fully flat: `{0,...,|S|-1}` (the bare leaf `|S|` always coarsens `S`)
- top-level-only: sizes of `S`'s outer slots, ignoring internal nesting

For my fold-1 shape `((2,2),2)`: `Z(((2,2),2)) = {Z_8, Z_(4,2),
Z_((2,2),2)}` — exactly the fully-flat (`k=5`), partly-combined
(`(r,i1)=(2,1)`), and fully-refined (`(i0,i1,i2)=(0,1,1)`) names for the
same box `f`.

If `P ⪯ S`, then `Z(P) ⊆ Z(S)` — follows directly from `⪯` being
transitive (anything coarsening `P` also coarsens `S`).

**Naming vs. computing an address are separate questions.** Fold-2's shape
`(2,(2,2))` still has `(2,4)` on its list of valid coordinate names (top
sizes are 2 and 4) — even though Figure 1 showed no flat *stride* exists
for it. A coordinate is just a legal name; whether that name comes with a
one-number way to compute a memory offset is a separate, later concern.

### 2.2.2 Coordinates (Def 2.9)

A coordinate for `S` is any value drawn from *any one* scheme in `Z(S)` —
not just the fully-refined one. Coordinates are `HTuple(N)` (natural
numbers, can be 0) — distinct from shapes, which are `HTuple(Z⁺)` (sizes
can't be 0).

Being "in-bounds" needs **two** checks, both required:
1. matches some scheme's rank/nesting (right *shape*)
2. every value sits inside that scheme's actual range (right *magnitude*)

Worked on `((2,2),2)` (all three modes have size 2, so refined coordinates
only ever range over `{0,1}`):
- `3` ✓ — valid in the fully-flat scheme (`0..7`)
- `(0,1)` ✓ — valid in the partly-combined scheme (`r<4, i1<2`)
- `(1,2,0)` ✗ — right shape (a triple), but `i1=2` overshoots its range
- `(1,0,2)` ✗ — same failure, `i2=2` overshoots

**Lesson:** matching the nesting profile is not the same as being in
range — both a real coordinate needs both.

### 2.2.2 (cont'd) — Integral/natural coordinates, idx2crd, admissibility

**Integral coordinate (2.10):** a coordinate drawn from `Z_|S|` — always a
bare flat number. E.g. `13` for shape `(4,20)`.

**Natural coordinate (2.11):** a coordinate drawn from `Z_S` — matches `S`'s
*exact* nesting, congruent to `S`. For `S=((2,2),2)`, the natural
coordinate has to follow the same brackets: `((i0,i2), i1)`, not a flat
triple.

**Trap caught by hand:** `(1,1,1)` looks plausible but is **not** admissible
for `((2,2),2)` — it's a flat triple (3 top-level slots) while `S` has only
2 top-level slots (`(2,2)` and `2`). Rank mismatch fails weak congruence
immediately, before any values are even checked. The plain shape `(2,2,2)`
and the grouped shape `((2,2),2)` total the same size (8) but have
different top-level splits (`2×2×2` vs `4×2`) — same trap as `2×15` vs
`6×5` from compatibility, one level up. The correctly-shaped natural
coordinate for box `d` (row=1, col=1, floor=0) is `((1,0), 1)`.

**idx2crd / crd2idx — my own day-one method, generalized.** `idx2crd` is
exactly `k → (i,j)` via mod/floor-div, extended to any number of modes
using the running product of preceding sizes as each step's divisor.
`crd2idx` is the reverse: multiply each coordinate by the running product
of everything before it, then sum — literally the "stride = running
product of preceding sizes" pattern I derived on my own back in 1.2.

Round-trip check on `(4,20)`: `idx2crd(13) = (1,3)`, and
`crd2idx((1,3)) = 1 + 3×4 = 13`. Also: `crd2idx((3,19)) = 3 + 19×4 = 79 =
|S|-1` — the last coordinate always maps to the last flat index.

**Admissible (2.12) vs. out-of-bounds (2.13):**
- admissible = right *profile* (weakly congruent to `S`) — doesn't check
  actual value ranges yet
- out-of-bounds = admissible, but the values overshoot their actual range
  (admissible minus in-bounds)

Worked on the **flat** shape `(2,2,2)` (three separate modes, each size 2
— not the grouped `((2,2),2)`, per the trap above): `(1,2,0)` and `(1,0,2)`
are both admissible (correct triple profile, congruent to `(2,2,2)`) but
out-of-bounds — `i1=2` and `i2=2` respectively overshoot the actual range
`{0,1}` for that shape.

**Lesson, stated plainly:** checking a coordinate needs three things in
order — (1) does it match some shape's *nesting* (admissible/weakly
congruent), (2) is that shape actually the one in question (or a real
member of `Z(S)`), (3) are the *values* in range (in-bounds). Skipping
step 2 is exactly what made `(1,1,1)` look valid when it wasn't.
