# CUTE Layout Representation and Algebra — Study Notes

Working through Cecka's CUTE preprint and Colfax's categorical-foundations
material, page by page, with my own worked examples.

## Progress
| Section | Status | Last updated |
|---|---|---|
| 1.2 Canonical loops | done | 2026-08-24 |
| 1.3 Tensors and folding | in progress | — |
| 2.1 Tuples and HTuples | not started | — |
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
Batch mode: present in A, B, and C. 

## 2. Layout representation
...