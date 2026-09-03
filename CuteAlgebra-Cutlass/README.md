# CUTE Layout Representation and Algebra — Study Notes

Working through Cecka's CUTE preprint and Colfax's categorical-foundations
material, page by page, with my own worked examples.

## Progress
| Section | Status | Last updated |
|---|---|---|
| 1.2 Canonical loops | done | 2026-08-24 |
| 1.3 Tensors and folding | started | — |
| 2.1 Tuples and HTuples | started | — |
...

## 1. Introduction and motivation
Spoke with Xintong (a Member of Technical Interns at Thinky Machines) and he mentioned the usefulness of Cutlass, Cute Algbrea and Layouts so I thought why not schedule some time to study this on the side as I continue writing my Flash Attention: Forwards and Backwards. 

### 1.2 Canonical loops and loop transformations
**Core idea:** a canonical loop nest is fully characterized by Shape:Stride. A linearized layout of a matrix layout is the matrix's codomain while the matrix is the domain. 

They have a relationship: if we have a loop say for (int m = 2; m <= 16; m+=3)...this loop has a relationship with another loop for (int i = 0;). We can understand the progression of i through the expression g(m) = f(i) = f(start + step * i). 

g(m) = f(0) = f(2 + 3 * 0) = g(2). 
g(m) = f(1) = f(2 + 3 * 1) = g(5).
g(m) = f(2) = f(2 + 3 * 2) = g(8).

**Why it matters:** foreshadows that transformations P are themselves layouts: This continues to show the relationship between both loops and that a transformation of a certain shape : stride is an object of another shape : stride. 

### 1.3 Tensors and folding
Simple idea: It shows the different modes in operands before and after doing through the multiplication operation. 
Row mode : this is present in A, absent in B, and present in C. 
column mode : this is absent in A, present in B, and present in C. 
reduction mode: present in A, present in B, and absent in C. 
Batch/floor/width mode: present in A, B, and C. 

Say we have a physical data of size 8 with each index containing letters a - h. We can describe it as a rank-3 tensor of a shape 2 x 2 x 2 with strides (2, 1, 4). 2 is the row step (mode 0), 1 is column step (mode 1), and 4 is the batch step (mode 2). 

If we fold mode 2 into mode 0, we can retain the step size, but with different layout but with the same strides declared differently. We'll get a 4 x 2 tensor of strides (2,1) but cute represenation ((2,2), 2) and strides ((2,4), 1). 

It's quite different if we fold mode 2 into mode 1 because walking through the four glued offsets gives jumps of +1, +3, +1, not a constant step, which is precisely why no single flat number can exist. However, Cute Algebra provides us with an elegant way of representing problematic tensors like that: we get a 2 x 4 tensor of representation (2, undefined) but with a cute representation of (2, (2,2)) and strides of (2, (1,4)). 

## 2. Layout representation
### Tuples 
A tuple, Tuple(T), is an ordered list of elements, all drawn from the same set T, where position (not differing properties) is what distinguishes one entry from another — say for a shape, T = Z⁺ (positive integers). Two operations: rank(X) is how many modes (dimensions) the tuple has hence it's the length(n) of the tuple; and accessor X_i is the entry at mode i (for 0 ≤ i < rank(X)).
Example: my fold-1 shape (4,2) is a Tuple(Z⁺) with rank((4,2)) = 2, X_0 = 4 (row), X_1 = 2 (column).

### HTuples 
These are almost like Tuples, but can be nested. This introduces another characterisitic called DEPTH.

All the below conditions hold if X is a tuple. 

Rand(X) of an HTuple is it's length(n) the numbers of elements in the tuples. If it's just one element, then the rank is 1. Xi is the location of each entry in the tuple as we start counting for zero (hence the math notation 0 less than or equal to X and X less than or equal to X's rank). 

Then Depth: this is 1 + max(depth(X0), depth(X1), ..., depth(Xn-1)), if we have only one element then the depth is 1. 

Examples: rank(((4,6),(3,(2,2),8))) has rank 2. It's depth is 3 and determined thus: depth = 3. The deepest chain is: root (3,(2,2),8) to (2,2) to ordinary numbers. That's 3 levels of "tuple containing a tuple" before one hits a plain number, which is exactly why it beats out the other branch, (4,6), which only goes one level deep. Depth tracks the worst-case nesting, not the average.

depth(((2,2), 2)): depth = 2. root (2,2) to bare numbers, that's the deep side; the other slot, plain 2, is already a leaf. 1 + max(1, 0) = 2.

...