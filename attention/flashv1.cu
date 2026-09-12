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


  
  