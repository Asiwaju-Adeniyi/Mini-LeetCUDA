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
auto smemLayoutQ =
    tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandA>{}, tileShapeQ);
Layout gmemLayoutQ =
    make_layout(make_shape(qRows, headDim, heads, batch),
                make_stride(headDim * heads, 1, headDim, heads * qRows * headDim));
Tensor qGmemTensor = make_tensor(qGlobal, gmemLayoutQ);
auto tmaQ =
    make_tma_copy(SM90_TMA_LOAD{}, qGmemTensor, smemLayoutQ, tileShapeQ, Int<1>{});

    auto tileShapeK = make_shape(TileK{}, TileD{});
    auto smemLayoutK = tile_to_shape(GMMA::Layout_K_SW128_Atom<OperandB>{}, tileShapeK);
    Layout gmemLayoutK = make_layout(make_shape(kRows, headDim, heads, batch), 
    make_stride(1, headDim, headDim * heads, kRows * headDim * heads));
    Tensor kGmemTensor = make_tensor(kGlobal, gmemLayoutK); 
    auto tmaK = make_tma_copy(SM90_TMA_LOAD{}, kGmemTensor, smemLayoutQ, tileShapeK, Int<1>);

}




  
  