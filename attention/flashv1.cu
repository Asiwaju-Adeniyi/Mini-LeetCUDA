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
   

};




