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


    auto tileShapeQ = make_shape(TileQ{}, TileD{});
    auto smemLayoutQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandA>{}, tileShapeQ);
    Layout gmemLayoutQ = make_layout(make_shape(qRows, headDim, heads, batch),
    make_stride(headDim * heads, 1, headDim, heads * qRows * headDim));
    Tensor qGmemTensor = make_tensor(qGlobal, gmemLayoutQ);
    auto tmaQ =make_tma_copy(SM90_TMA_LOAD{}, qGmemTensor, smemLayoutQ, tileShapeQ, Int<1>{});

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

    #ifdef CTA256
    using WarpgroupCount = Layout<Shape<_2, _1, _1>>;
    #else
    using WarpgroupCount = Layout<Shape<_1, _1, _1>>;
    #endif
   
    using TiledMmaGemm1 = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<OperandA, OperandB, 
      Accumulator, Shape<TileQ, TileK, TileD>>(), WarpgroupCount{}));

  #ifdef SINSEM 
  using TileMmaGemm2 = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<OperandA, OperandB, Accumulator, Shape<TileQ, 
  TileD, TileK, GMMA::Major::K, GMMA::Major::MN>(), WarpgroupCount{}));
  #else
  using TileMmaGemm2 = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<OperandA, OperandB, Accumulator, 
    Shape<TileQ, TileD, TileK, GMMA::Major::K, GMMA::Major::MN>(), WarpgroupCount{}));
  #endif 
}
  
  