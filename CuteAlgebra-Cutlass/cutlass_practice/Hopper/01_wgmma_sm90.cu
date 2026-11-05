// Hopper WGMMA track.
//
// Following Colfax's CUTLASS tutorial series:
//   Part 1 - WGMMA: https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
//   TMA:    https://research.colfax-intl.com/tutorial-hopper-tma/
//   Part 2 - pipelining: https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
//
// Scaffold only. The first kernel to implement here is a single-stage SM90
// WGMMA GEMM; TMA and pipelining follow in later files.
