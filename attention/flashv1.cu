template <typename StorageT, typename AccumT, int HeadDimCT>
void fmhaForwardDevice(int numQueries, int numKeys, int numHeads, int batchSize,
                       StorageT const *qGlobal, StorageT const *kGlobal,
                       StorageT *vGlobal, StorageT *sGlobal, StorageT *oGlobal,
                       AccumT *rowMaxOut, AccumT *rowSumOut, int iterations,
                       float scale, cudaStream_t stream = 0) {
  using namespace cute;

  // runtime problem sizes
  auto batch    = int(batchSize);
  auto heads    = int(numHeads);
  auto qRows    = int(numQueries);
  auto kRows    = int(numKeys);
  auto headDim  = int(HeadDimCT);

  // compile-time tile sizes
  using TileQ = Int<kQueriesPerBlock>;
  using TileK = Int<kKeysPerBlock>;
  using TileD = Int<HeadDimCT>;            

  using OperandA = StorageT;
  using OperandB = StorageT;
  using Accumulator = AccumT;
  using ClusterShape = Shape<_1, _1, _1>;


    auto tileShapeQ = make_shape(TileQ{}, TileD{});
    auto smemLayoutQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandA>{}, tileShapeQ);
    Layout gmemLayoutQ = make_layout(make_shape(qRows, headDim, heads, batch),
    make_stride(headDim * heads, 1, headDim, heads * qRows * headDim));
    Tensor qGmemTensor = make_tensor(qGlobal, gmemLayoutQ);
    auto tmaQ =make_tma_copy(SM90_TMA_LOAD{}, qGmemTensor, smemLayoutQ, tileShapeQ, Int<1>{});

    auto tileShapeO = make_shape(TileQ{}, TileD{});
    Layout gmemLayoutO = make_layout(make_shape(qRows, headDim, heads, batch),
    make_stride(headDim*heads, 1, headDim, heads*qRows*headDim));
    Tensor oGmemTensor = make_tensor(oGlobal, gmemLayoutO);   // fixed: gmemLayoutO, not gmemLayoutQ
    auto tmaO = make_tma_copy(SM90_TMA_STORE{}, oGmemTensor, smemLayoutQ, tileShapeO, Int<1>{});

    auto tileShapeK = make_shape(TileK{}, TileD{});
    auto smemLayoutK = tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandB>{}, tileShapeK);
    Layout gmemLayoutK = make_layout(make_shape(kRows, headDim, heads, batch), 
    make_stride(headDim * heads, 1, headDim, kRows * headDim * heads));
    Tensor kGmemTensor = make_tensor(kGlobal, gmemLayoutK); 
    auto tmaK = make_tma_copy(SM90_TMA_LOAD{}, kGmemTensor, smemLayoutK, tileShapeK, Int<1>{});
    

    auto tileShapeV = make_shape(TileK{}, TileD{});
    auto smemLayoutV = tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandB>{}, tileShapeV);
    Layout gmemLayoutV = make_layout(make_shape(kRows, headDim, heads, batch), 
    make_stride(headDim * heads, 1, headDim, kRows * headDim * heads));
    Tensor vGmemTensor = make_tensor(vGlobal, gmemLayoutV);
    auto tmaV = make_tma_copy(SM90_TMA_LOAD{}, vGmemTensor, smemLayoutV, tileShapeV, Int<1>{});

    auto tileShapeVt = make_shape(TileD{}, TileK{});
    auto smemLayoutVt = composition(smemLayoutV, make_layout(tileShapeVt, GenRowMajor{}));

    auto tileShapeS = make_shape(TileQ{}, TileK{});
    Layout gmemLayoutS = make_layout(make_shape(qRows, kRows, heads, batch),
    make_stride(kRows, 1, kRows*qRows, heads*qRows*kRows));
    auto smemLayoutS = tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandA>{}, tileShapeS);

    #ifdef CTA256
    using WarpgroupCount = Layout<Shape<_2, _1, _1>>;
    #else
    using WarpgroupCount = Layout<Shape<_1, _1, _1>>;
    #endif
   
    using TiledMmaGemm1 = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<OperandA, OperandB, 
      Accumulator, Shape<TileQ, TileK, TileD>>(), WarpgroupCount{}));

    #ifdef SINSMEM 
    using TileMmaGemm2 = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<OperandA, OperandB, Accumulator, Shape<TileQ, 
    TileD, TileK, GMMA::Major::K>, GMMA::Major::MN>(), WarpgroupCount{}));
    #else
    
    using TiledMmaGemm2 = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<OperandA, OperandB, Accumulator, 
      Shape<TileQ, TileD, TileK>, GMMA::Major::K, GMMA::Major::MN>(), WarpgroupCount{}));
    #endif 

//Launch configs copied from Colfax's implementation
  
void const *kernel = (void const *)fmhaForward<
    StorageT, AccumT, TiledMmaGemm1, TiledMmaGemm2, decltype(tmaQ),
    decltype(tileShapeQ), decltype(gmemLayoutQ), decltype(smemLayoutQ),
    decltype(tmaK), decltype(tileShapeK), decltype(gmemLayoutK), decltype(smemLayoutK),
    decltype(tileShapeS), decltype(gmemLayoutS), decltype(smemLayoutS),
    decltype(tmaV), decltype(tileShapeV), decltype(gmemLayoutV), decltype(smemLayoutV), decltype(smemLayoutVt),
    decltype(tmaO), decltype(tileShapeO), decltype(gmemLayoutO),
    decltype(gmemLayoutMi), ClusterShape>;

auto smem_size = int(sizeof(SharedStorage<OperandA, decltype(smemLayoutQ), decltype(smemLayoutK),
                            decltype(smemLayoutS), decltype(smemLayoutV)>));
cfk::utils::set_smem_size(smem_size, kernel);

dim3 block_dims(size(TiledMmaGemm1{}));
dim3 grid_dims(ceil_div(size(qRows), size(TileQ{})), heads, batch);
dim3 cluster_dims(size<0>(ClusterShape{}), 1, 1);

cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims, smem_size, stream};
auto nTilesOfK = ceil_div(size(kRows), size(TileK{}));

for (int i = 0; i < iterations; ++i) {
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, kernel, qGlobal, tmaQ, tileShapeQ, gmemLayoutQ, smemLayoutQ,
      kGlobal, tmaK, tileShapeK, gmemLayoutK, smemLayoutK,
      sGlobal, tileShapeS, gmemLayoutS, smemLayoutS, nTilesOfK,
      vGlobal, tmaV, tileShapeV, gmemLayoutV, smemLayoutV, smemLayoutVt,
      oGlobal, tmaO, tileShapeO, gmemLayoutO,
      rowMaxOut, rowSumOut, gmemLayoutMi, scale);
}

}

template <class ElementType, class SmemLayoutQ, class SmemLayoutK,
          class SmemLayoutS, class SmemLayoutV>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
#ifdef SINSMEM
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutS>> smem_s;
#endif
  cute::uint64_t tma_load_mbar[8];

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), smemLayoutQ);
Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), smemLayoutK);
#ifdef SINSMEM
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem_s.data()), smemLayoutS);
#else
  // Just a dummy sS (with smem_v). It's required only for shape later.
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutS);
#endif
Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutV);
Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutVt);

Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));

TiledMma0 tiledMma0;
auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);

//signature, shared memory, the TMA-aware tensors, per-thread MMA slices

template <class StorageT, class AccumT, class TiledMmaGemm1, class TiledMmaGemm2,
          class TiledCopyQ, class TileShapeQ, class GmemLayoutQ, class SmemLayoutQ,
          class TiledCopyK, class TileShapeK, class GmemLayoutK, class SmemLayoutK,
          class TileShapeS, class GmemLayoutS, class SmemLayoutS,
          class TiledCopyV, class TileShapeV, class GmemLayoutV, class SmemLayoutV, class SmemLayoutVt,
          class TiledCopyO, class TileShapeO, class GmemLayoutO,
          class GmemLayoutMi, class ClusterShape>
__global__ static void
fmhaForward(StorageT const *qGlobal, TiledCopyQ const tmaQ, TileShapeQ tileShapeQ,
            GmemLayoutQ gmemLayoutQ, SmemLayoutQ smemLayoutQ,
            StorageT const *kGlobal, TiledCopyK const tmaK, TileShapeK tileShapeK,
            GmemLayoutK gmemLayoutK, SmemLayoutK smemLayoutK,
            StorageT *sGlobal, TileShapeS tileShapeS, GmemLayoutS gmemLayoutS,
            SmemLayoutS smemLayoutS, int nTilesOfK,
            StorageT *vGlobal, TiledCopyV const tmaV, TileShapeV tileShapeV,
            GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV, SmemLayoutVt smemLayoutVt,
            StorageT *oGlobal, TiledCopyO const tmaO, TileShapeO tileShapeO, GmemLayoutO gmemLayoutO,
            AccumT *rowMaxOut, AccumT *rowSumOut, GmemLayoutMi gmemLayoutMi, float scale) {
  using namespace cute;

  extern __shared__ char shared_memory[];
  using SharedStorageT = SharedStorage<StorageT, SmemLayoutQ, SmemLayoutK, SmemLayoutS, SmemLayoutV>;
  SharedStorageT &shared_storage = *reinterpret_cast<SharedStorageT *>(shared_memory);
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;

  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), smemLayoutQ);
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), smemLayoutK);
#ifdef SINSMEM
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem_s.data()), smemLayoutS);
#else
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutS); // dummy, shape only
#endif
  Tensor sV  = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutV);
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutVt);

  Tensor mQ = tmaQ.get_tma_tensor(shape(gmemLayoutQ));
  Tensor mK = tmaK.get_tma_tensor(shape(gmemLayoutK));
  Tensor mV = tmaV.get_tma_tensor(shape(gmemLayoutV));
  Tensor mO = tmaO.get_tma_tensor(shape(gmemLayoutO));

  TiledMmaGemm1 tiledMma0;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  TiledMmaGemm2 tiledMma1;
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

  // Cluster/multicast bookkeeping — inert given ClusterShape = Shape<_1,_1,_1>.
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape{});
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                   block_rank_in_cluster / cluster_shape_x};
  uint16_t mcast_mask_a = 0;
  auto block_layout = Layout<ClusterShape>{};
  for (int n = 0; n < size(block_layout); ++n)
    mcast_mask_a |= (uint16_t(1) << block_layout(n, 0, Int<0>{}));

  auto cta_tmaQ = tmaQ.get_slice(0);
  auto cta_tmaK = tmaK.get_slice(cluster_local_block_id.x);
  auto cta_tmaV = tmaV.get_slice(cluster_local_block_id.x);
  auto cta_tmaO = tmaO.get_slice(0);



  //Q's tiling, register-fragment allocation for both GEMMs, barrier setup, the first Q load:
   

    auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gQ = local_tile(mQ, tileShapeQ, blkCoordQ);

  Tensor tQgQX = cta_tmaQ.partition_S(gQ);
  Tensor tQgQ  = group_modes<1, rank(tQgQX)>(tQgQX);
  auto kTiles = size<1>(tQgQ);
  assert(kTiles == 1);
  assert(kTiles == size<2>(gQ));

  Tensor tQsQX = cta_tmaQ.partition_D(sQ);
  Tensor tQsQ  = group_modes<1, rank(tQsQX)>(tQsQX);
  Tensor tKsKX = cta_tmaK.partition_D(sK);
  Tensor tKsK  = group_modes<1, rank(tKsKX)>(tKsKX);
  Tensor tVsVX = cta_tmaV.partition_D(sV);
  Tensor tVsV  = group_modes<1, rank(tVsVX)>(tVsVX);
  static_assert(size<1>(tQsQ) == 1);
  static_assert(size<1>(tKsK) == 1);

  // GEMM-I fragments.
  Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
  Tensor tSrK = threadMma0.partition_fragment_B(sK);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
  clear(tSrS);

  // GEMM-II fragments (S becomes P).
  Tensor tOrV = threadMma1.partition_fragment_B(sVt);
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);

#ifdef SINSMEM
  Tensor tSsS = threadMma0.partition_C(sS);
  cute::fill(tSsS, StorageT(0.0));
  Tensor tOrP = threadMma1.partition_fragment_A(sS);
#else
  Tensor tOrS = threadMma1.partition_fragment_A(sS);
  auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);
  auto tOrP = make_tensor(tSrS.data(), tOrPLayout);
#endif

  Tensor rowMax = make_tensor<AccumT>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<AccumT>::infinity());
  cute::fill(rowSum, AccumT(0.0));

  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();

  cfk::barrierInit(tma_load_mbar[0], 1); // K
  cfk::barrierInit(tma_load_mbar[1], 1); // V
  cfk::barrierInit(tma_load_mbar[2], 1); // Q

  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaQ, tma_load_mbar[2]);
  cute::wait_barrier(tma_load_mbar[2], 0); // required

  // the actual mainloop: prefetch, GEMM-I, online softmax, GEMM-II, phase flip

     auto blkCoordK = make_coord(0, 0, blockIdxH, blockIdxB);
  Tensor gK = local_tile(mK, tileShapeK, blkCoordK);
  Tensor tKgKX = cta_tmaK.partition_S(gK);
  Tensor tKgK  = group_modes<1, rank(tKgKX)>(tKgKX);
  assert(size<1>(tKgK) == size<2>(gK));
  assert(size<1>(tKgK) == kTiles);

  cfk::copy(tKgK(_, 0), tKsK(_, 0), tmaK, tma_load_mbar[0], mcast_mask_a);
  int phase = 0;

#pragma unroll
  for (uint64_t blockIdxY = 0; blockIdxY < nTilesOfK; ++blockIdxY) {

    auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
    Tensor gV = local_tile(mV, tileShapeV, blkCoordV);
    Tensor tVgVX = cta_tmaV.partition_S(gV);
    Tensor tVgV  = group_modes<1, rank(tVgVX)>(tVgVX);

    cfk::syncCluster<ClusterShape>();
    cfk::copy(tVgV(_, 0), tVsV(_, 0), tmaV, tma_load_mbar[1], mcast_mask_a);
    clear(tSrS);

    cfk::gemm_ldbar(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0], phase); // GEMM-I

#ifdef COPYOUTMM0  // verification-only, matches sGlobal debug path
    Tensor mS = make_tensor(make_gmem_ptr(sGlobal), gmemLayoutS);
    auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
    Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
    Tensor tSgS = threadMma0.partition_C(gS);
    copy(tSrS, tSgS);
#endif

    if (blockIdxY != (nTilesOfK - 1)) {
      auto blkCoordKNext = make_coord(blockIdxY + 1, 0, blockIdxH, blockIdxB);
      auto gKNext = local_tile(mK, tileShapeK, blkCoordKNext);
      Tensor tKgKNextX = cta_tmaK.partition_S(gKNext);
      Tensor tKgKNext  = group_modes<1, rank(tKgKNextX)>(tKgKNextX);
      cfk::syncCluster<ClusterShape>();
      cfk::copy(tKgKNext(_, 0), tKsK(_, 0), tmaK, tma_load_mbar[0], mcast_mask_a);
    }

    if (blockIdxY == 0) {
      onlineSoftmaxAndRescale<true, AccumT>(rowMax, rowSum, tSrS, tOrO, scale);
    } else {
      onlineSoftmaxAndRescale<false, AccumT>(rowMax, rowSum, tSrS, tOrO, scale);
    }
    warpgroup_fence_operand(tSrS);

#ifdef SINSMEM
    cfk::copy(tSrS, tSsS);
    cfk::gemm_ldbar(tiledMma1, tOrP, tOrV, tOrO, tma_load_mbar[1], phase);
#else
    cfk::gemm_ldbar(tiledMma1, convert_type<StorageT, AccumT>(tOrP), tOrV,
                    tOrO, tma_load_mbar[1], phase);
#endif
    phase = (phase + 1) % 2;
  }

};




