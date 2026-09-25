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
};


