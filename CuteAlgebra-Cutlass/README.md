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
| 2.2.1 Coordinate Sets/Compatibility | done |  |
| 2.2.2 Coordinates | done |  |
| 2.3 Stride | done |  |
| 2.3.1 Integer Semimodules | done |  |

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

### 2.3 Stride (Def 2.15)

A stride is just a set of step-sizes, one per mode — and it must be
**congruent** to its shape (Def 2.3): same nesting brackets, leaves filled
with step-sizes instead of extents. My fold-1 `S=((2,2),2)` and
`D=((2,4),1)` qualify by construction.

`inner_product` is my own offset formula (`2×i0+1×i1+4×i2`, used since page
one), made recursive: base case is plain multiply (`c·d`); the HTuple case
recurses slot-by-slot and **sums** the results —
`Σᵢ inner_product(cᵢ,dᵢ)`. Handles nested slots; reduces to the same plain
arithmetic when everything's flat.

**Verified on two boxes**, `S=((2,2),2)`, `D=((2,4),1)`:
- box `e` (row0,col0,floor1), natural coord `((0,1),0)`:
  `inner_product((0,1),(2,4)) + 0×1 = (0×2+1×4) + 0 = 4` ✓
- box `g` (row1,col0,floor1), natural coord `((1,1),0)`:
  `(1×2+1×4) + 0×1 = 6` ✓

Both match the actual physical offsets — confirms the recursive formula is
just the old arithmetic, generalized to handle nesting.

*Note for later: strides don't have to be plain integers — `D` can be any
"integer-semimodule" (e.g. coordinate-valued or XOR-valued), which is what
enables swizzling later. Not needed yet.*

### 2.3.1 Integer-semimodules (Def 2.16)

Plain meaning: a set of things you're allowed to (1) **add together** and
(2) **scale by a whole number**. Plain integers already qualify — you can
add them and scale them. Pairs of numbers qualify too, using the same two
operations, just componentwise.

**e0 and e1** aren't derived from anything — they're just two hand-picked
building-block pairs: `e0=(1,0)`, `e1=(0,1)`. The claim: *any* pair can be
built from some number of copies of each, added together.

My own check: build `(4,7)`.
- `4×e0 = 4×(1,0) = (4,0)`
- `7×e1 = 7×(0,1) = (0,7)`
- `(4,0) + (0,7) = (4,7)` ✓ — 4 copies of `e0`, 7 copies of `e1`.

**Why this matters for strides:** if a stride is built from `e0`/`e1`
instead of plain numbers, `inner_product` no longer computes a memory
offset — it hands back a *coordinate*. Worked example, shape `(3,2)`,
stride `D=(e0, 2e1)`, coordinate `(2,1)`:

`inner_product((2,1),(e0,2e1)) = 2·e0 + 1·(2e1) = (2,0) + (0,2) = (2,2)`

Input `(2,1)` → output `(2,2)` — the `2e1` stride stretched the second
slot by 2. (`D=(e0,e1)` alone would've echoed the input back unchanged —
the "identity" case.)
### 2.3.1 (cont'd) — F2 = ({0,1}, XOR, AND)

A tiny integer-semimodule: only two elements, `0` and `1`.

- **"Adding" = XOR** — no mismatch to fix, since both inputs are always
  already `0` or `1`. Checked associativity on `1,1,1`: `1⊕(1⊕1) = 1⊕0 = 1`
  and `(1⊕1)⊕1 = 0⊕1 = 1` — same either way.
- **"Scaling" = AND, but needs a shrink step first.** Scaling has a
  mismatch: the scalar is *any* whole number, not automatically a member
  of `{0,1}`. Fix: shrink the whole number to `0` or `1` by odd/even
  (mod 2) *before* applying AND. Odd → shrinks to `1`. Even → shrinks to
  `0`.

Worked: `3·1` → 3 is odd → shrinks to `1` → `AND(1,1)=1`. `4·1` → even →
shrinks to `0` → `AND(0,1)=0`. `5·1=1`. `6·0` → even → shrinks to `0` →
`AND(0,0)=0`.

**Why XOR never needs the shrink step:** both operands of an addition are
*already* elements of `{0,1}` — never an outside whole number — so there's
no mismatch to fix. Checked directly: `0⊕1⊕1` → `0⊕1=1`, `1⊕1=0`, never
left `{0,1}` at any step.

**Why it matters:** this is the same F2/XOR machinery behind Figure 1's
"binary swizzle" example (`f1,f5,f16` strides) — the tool CUTE uses to
describe scrambled, bank-conflict-avoiding shared-memory access patterns.

### 2.4 Layout (Def 2.17)

A layout is a two-step machine: **step 1 (shape)** converts a plain number
into a coordinate (`idx2crd`/`crd2idx`, my day-one `k↔(i,j)` trick).
**step 2 (stride)** converts that coordinate into an output via
`inner_product`. `L = D∘S` — read right to left: shape runs first, then
stride.

**Worked on shape `(4,20)`, two different strides:**

| k | step 1: (row,col) | stride `(1,4)` (col-major) | stride `(20,1)` (row-major) |
|---|---|---|---|
| 13 | (1,3) | `1×1+3×4=13` | `1×20+3×1=23` |
| 25 | (1,6) | `1×1+6×4=25` | `1×20+6×1=26` |

`(1,4)` gives back the input exactly — it's the shape's own natural
prefix-product stride, so shape-then-stride cancels out (same identity
phenomenon as the `e0,e1` example). `(20,1)` genuinely transforms it.

**The actual point:** the *shape* `(4,20)` never decided row-major vs.
column-major — the *stride* did, entirely. Same shape, different strides,
different memory behavior. This is exactly what shows up in CUTLASS
FlashAttention kernels: shared-memory layouts for WGMMA/TMA often use
neither plain row-major nor column-major, but whatever custom stride the
hardware instruction demands.

### 2.4.1 Notations and Operations

Three ways to write the same layout: `S/D` (fraction), `S:D` (colon),
`D∘S` (composition — right-to-left, matching the two-step pipeline: `S`
first, `D` second).

**Every layout property below comes straight from the shape — stride is
irrelevant to all of them; it only matters once you actually run
`inner_product`:**
- `rank(L) = rank(S)`, `depth(L) = depth(S)`, `|L| = |S|`
- `L_i = S_i : D_i` — pure positional pairing, no computation: pair slot
  `i` of the shape with slot `i` of the stride, nesting intact
- `Z(L) = Z(S)` — valid coordinate names come from the shape alone
- `L∼U ⇔ S∼X` and `L⪯U ⇔ S⪯X` — congruence/compatibility of layouts
  reduces entirely to congruence/compatibility of their shapes

**Checked on my own fold-1 and fold-2 layouts:**
- `((2,2),2):((2,4),1)` → `L_0 = (2,2):(2,4)`
- `(2,(2,2)):(2,(1,4))` → `rank(L)=2` (top-level slots are `2` and `(2,2)`
  — not 3, same rank-counting rule from HTuples), `depth(L)=2`, `L_0=2:2`,
  `L_1=(2,2):(1,4)`

**Takeaway:** stride affects what a layout *computes*; never what it's
*shaped like*.

### 2.4.2 Layout Examples

Six layouts, same shape `(4,8)`, different strides — proof that `offset =
Σ coordinate × stride` covers everything from trivial to genuinely novel.

**(a)-(c) col-major/row-major/padded** — nothing new, just my own
`inner_product` work relabeled. Checked `(1,2)`: `(1,4)`→`9`, `(8,1)`→`10`.
Padding is just bumping the column step (4→5) to leave gaps.

**(d) Col-Major Interleave**, `(4,(4,2)):(4,(1,16))` — column axis splits
into two chunks: `n0<4` (fast, step 1), `n1<2` (slow, step 16). Checked
`(0,(0,1))`: `0+0+16=16` ✓. Physically: first 4 logical columns sit packed
together, then jump to a separate chunk starting at 16 — two contiguous
regions instead of one smooth row (e.g. splitting a head-dim into tiles).

**(e) Mixed**, `((2,2),(4,2)):((1,8),(2,16))` — *both* axes split at once.
Checked `((1,0),(2,0))`: `1+0+4+0=5` ✓. This is the general shape of real
hardware tile-partitioning (thread/value layouts for tensor-core
instructions, coming up formally in Section 3.3.4).

**(f) Blocked Broadcast**, `((2,2),(2,4)):((0,2),(0,4))` — **stride can be
0.** `m0` and `n0` both have stride 0, so they drop out of
`inner_product` entirely — several different coordinates produce the exact
same offset on purpose. Checked `((1,1),(1,3))`: `1×0+1×2+1×0+3×4=14` ✓,
and `m0,n0`'s actual values were irrelevant to the result. Breaks the
assumption that layouts are one-to-one — this is how broadcasting one
value across a tile is expressed. (Same layout reappears later as a
"stride-0 modes don't contribute" example for complement, Section 3.5.)

**Figure 4 preview (not detailed yet):** strides built from the
non-integer semimodules from 2.3.1 — `e0/e1`-style strides produce
coordinate-outputting layouts (useful for bounds-checking, TMA
instructions); XOR/`F2`-style strides produce swizzle patterns for
avoiding shared-memory bank conflicts. Already have both underlying pieces
from 2.3.1 — this section is just where they get applied.

### 2.4.3 Completeness

Claim: **any** function on a finite domain, as long as `f(0)=0`, can be
built by chaining CUTE layouts together via composition — not just
row-major/padding/swizzle, literally any function reachable this way.

Why `f(0)=0` is required, not arbitrary: `inner_product` always sends the
all-zero coordinate to `0` — true at every single step of the recursion,
so every composition of layouts inherits it automatically. Completeness
can't cover a function that violates something every layout is
structurally incapable of violating.

**Takeaway:** CUTE isn't a convenient toolkit covering the common cases —
row-major, padding, interleave, swizzle, broadcast are all just familiar
instances of something provably maximally expressive.

Figure 4 previews strides from 2.3.1's non-integer semimodules in action:
`e0/e1`-style strides → coordinate-outputting layouts (identity, transpose;
useful for bounds-checking, TMA); XOR/`F2`-style strides → swizzle
patterns for shared-memory bank-conflict avoidance. Both underlying pieces
already covered in 2.3.1 — flagging the binary-swizzle example (`f1,f5,
f16`) as needing a proper revisit once actual kernel swizzling is
load-bearing, since it uses bit-permute stride notation not yet derived.

### 2.4.4 Semi-Linearity

`L(c) = d·c` (Eq. 7) once `c` is already a **natural** coordinate — this
is just `inner_product`, i.e. a dot product, and dot products are linear
(same superposition property as any linear operator from physics):
`d·(αc0+βc1) = α(d·c0)+β(d·c1)`.

**Where linearity breaks, shown with my own numbers — carrying.** Shape
`(4,20)`, `k0=3, k1=2`:
- convert-then-add: `idx2crd(3)+idx2crd(2) = (3,0)+(2,0) = (5,0)`
- add-then-convert: `idx2crd(3+2) = idx2crd(5) = (1,1)`

`(5,0) ≠ (1,1)` — mod/floor-div (the shape function) doesn't distribute
over addition, same phenomenon as carrying in multi-digit addition. So:
linear in natural coordinates, **not** linear in flat/arbitrary ones —
because only the stride half of a layout is linear; the shape half is
linear only when there's no conversion left to do.

**Matrix-vector framing:** `d·c = Dc`. Plain integer strides → `D` is a
`1×n` row (e.g. row-major `(20,1)` on `(4,20)`: `D=[20,1]`, checked
`D·(1,3)ᵀ = 20+3 = 23`, matches). Coordinate-valued strides (`e0,e1`) → `D`
is a genuine `m×n` matrix — `e0,e1` as columns gives the identity matrix,
which is exactly why that layout just echoes its input back.