#include <cassert>
#include <stdexcept>

class M16K16SharedMemLayout {
protected:
  __device__ constexpr static unsigned getIndex(unsigned m, unsigned k) {
    return (m << 4) | (m ^ k);
  }
};

// This class assumes blockDim.x == 32
class AM16K16MemCopy: protected M16K16SharedMemLayout {
public:
  // A is the base address of the original matrix A
  // curr_base_* is the starting coordinate of the current M16K16 block
  // M and K are the size of the original matrix A
  // warp_task_id is a per-warp integer within [0,8) representing the task id
  __device__ static double load(const double *A, bool a_rowmajor,
    std::size_t curr_base_m, std::size_t curr_base_k,
    std::size_t M, std::size_t K, std::size_t LDA, unsigned warp_task_id) {

    if (warp_task_id >= 8) return 0.0;
    unsigned pos1 = threadIdx.x & 15;
    unsigned pos2 = warp_task_id * 2u | (threadIdx.x >> 4);
    unsigned subm = a_rowmajor ? pos2 : pos1;
    unsigned subk = a_rowmajor ? pos1 : pos2;
    std::size_t m = subm + curr_base_m;
    std::size_t k = subk + curr_base_k;
    if (m >= M || k >= K) return 0.0;
    return __ldg(A + (a_rowmajor ? m : k) * LDA + (a_rowmajor ? k : m));
  }

  __device__ static void store(double *sha, double val,
    bool a_rowmajor, unsigned warp_task_id) {

    if (warp_task_id < 8) {
      unsigned pos1 = threadIdx.x & 15;
      unsigned pos2 = warp_task_id * 2u | (threadIdx.x >> 4);
      unsigned subm = a_rowmajor ? pos2 : pos1;
      unsigned subk = a_rowmajor ? pos1 : pos2;
      sha[getIndex(subm, subk)] = val;
    }
  }
};

template <unsigned MDiv16, unsigned NDiv16>
class WarpAccumulator: protected M16K16SharedMemLayout {
  /* for M16N16 block, each thread manages M4N2 8 elements */
  double m_acc[NDiv16][MDiv16][2][4];
  const unsigned m_startM, m_startN;
public:
  __device__ WarpAccumulator(): m_acc{}, m_startM((threadIdx.x & 3) << 2),
    m_startN((threadIdx.x & 0xFC) >> 1) {

    assert(blockDim.x == 32);
  }
  /* sha[MDiv16][16][16]
     shb[NDiv16][16][16] */
  __device__ void accK16(const double *sha, const double *shb) {
    for (unsigned kb = 0; kb < 16; kb += 2) {
      const double *a01_ = sha + getIndex(m_startM, kb);
      const double *a23_ = sha + getIndex(m_startM+2, kb);
      const double *b01_ = shb + getIndex(m_startN, kb);
      #pragma unroll
      for (unsigned ks = 0; ks < 2; ++ks) {
        double aelems[MDiv16][4];
        double belems[MDiv16][2];
        #pragma unroll
        for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
          aelems[md16][0] = a01_[getIndex(0, ks) + md16 * 256];
          aelems[md16][1] = a01_[getIndex(1, ks) + md16 * 256];
          aelems[md16][2] = a23_[getIndex(0, ks) + md16 * 256];
          aelems[md16][3] = a23_[getIndex(1, ks) + md16 * 256];
        }
        #pragma unroll
        for (unsigned nd16 = 0; nd16 < NDiv16; ++nd16) {
          belems[nd16][0] = b01_[getIndex(0, ks) + nd16 * 256];
          belems[nd16][1] = b01_[getIndex(1, ks) + nd16 * 256];
        }
        #pragma unroll
        for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
          #pragma unroll
          for (unsigned nd16 = 0; nd16 < NDiv16; ++nd16) {
            /* m16n16k1 calculation */
            /* each thread only do m4n2k1 */
            m_acc[nd16][md16][0][0] += aelems[md16][0] * belems[nd16][0];
            m_acc[nd16][md16][0][1] += aelems[md16][1] * belems[nd16][0];
            m_acc[nd16][md16][0][2] += aelems[md16][2] * belems[nd16][0];
            m_acc[nd16][md16][0][3] += aelems[md16][3] * belems[nd16][0];
            m_acc[nd16][md16][1][0] += aelems[md16][0] * belems[nd16][1];
            m_acc[nd16][md16][1][1] += aelems[md16][1] * belems[nd16][1];
            m_acc[nd16][md16][1][2] += aelems[md16][2] * belems[nd16][1];
            m_acc[nd16][md16][1][3] += aelems[md16][3] * belems[nd16][1];
          }
        }
      }
    }
  }
  /* C is row-major, eff_m <= MDiv16*16, eff_n <= NDiv16*16 */
  __device__ void storeAdd(double *C, unsigned eff_m, unsigned eff_n,
    std::size_t LDC) {

    auto swap2 = [](double &a, double &b)->void {
      double tmp = a;
      a = b;
      b = tmp;
    };
    auto swap8 = [&]()->void {
      if (threadIdx.x & 1) {
        #pragma unroll
        for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
          #pragma unroll
          for (unsigned nd16 = 0; nd16 < NDiv16; ++nd16) {
            #pragma unroll
            for (unsigned m = 0; m < 4; ++m) {
              swap2(m_acc[nd16][md16][0][m], m_acc[nd16][md16][1][m]);
            }
          }
        }
      }
    };
    swap8();
    const unsigned shflidx = threadIdx.x ^ 1;
    #pragma unroll
    for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
      #pragma unroll
      for (unsigned nd16 = 0; nd16 < NDiv16; ++nd16) {
        #pragma unroll
        for (unsigned m = 0; m < 4; ++m) {
          m_acc[nd16][md16][1][m] = __shfl_sync(0xFFFFFFFF,
            m_acc[nd16][md16][1][m], shflidx);
        }
      }
    }
    swap8();
    unsigned nbase = (threadIdx.x >> 2 << 1) | (threadIdx.x & 1);
    unsigned mbase = (threadIdx.x & 2) << 2;
    #pragma unroll
    for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
      unsigned mpos = mbase + md16 * 16u;
      #pragma unroll
      for (unsigned nd16 = 0; nd16 < NDiv16; ++nd16) {
        unsigned npos = nbase + nd16 * 16u;
        if (npos < eff_n) {
          #pragma unroll
          for (unsigned ms = 0; ms < 8; ++ms) {
            unsigned currm = mpos + ms;
            if (currm >= eff_m) break;
            C[currm * LDC + npos] += m_acc[nd16][md16][ms>>2][ms&3];
          }
        }
      }
    }
  }
};

__launch_bounds__(256, 1)
__global__ void dgemm_kernel(const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) {

  assert(blockDim.y == 2 && blockDim.z == 4);
  const std::size_t nblks = (N+95u) / 96u;
  const std::size_t mblkid = blockIdx.x / nblks;
  const std::size_t nblkid_off = blockIdx.x % nblks;
  const std::size_t nblkid = mblkid & 1u ?
    (nblks - 1u - nblkid_off) : nblkid_off;
  const std::size_t block_m_base = mblkid * 192u;
  const std::size_t block_n_base = nblkid * 96u;

  __shared__ double buffa[192 * 16], buffb[96 * 16];
  const unsigned calc_m_block_id = threadIdx.z;
  const unsigned calc_n_block_id = threadIdx.y;
  const std::size_t warp_calc_m_base = block_m_base + calc_m_block_id * 48u;
  const std::size_t warp_calc_n_base = block_n_base + calc_n_block_id * 48u;

  const bool calc_valid = warp_calc_m_base < M && warp_calc_n_base < N;
  WarpAccumulator<3,3> acc_m48n48;
  for (std::size_t k = 0; k < K; k += 16) {
    double copya[12], copyb[6];
    unsigned taskid = threadIdx.z * blockDim.y + threadIdx.y;
    #pragma unroll
    for (unsigned bid = 0; bid < 6; ++bid) {
      copya[bid] = AM16K16MemCopy::load(devA, a_rowmajor, block_m_base + bid * 16,
        k, M, K, LDA, taskid);
      copya[bid+6] = AM16K16MemCopy::load(devA, a_rowmajor, block_m_base + (bid+6) * 16,
        k, M, K, LDA, taskid);
      copyb[bid] = AM16K16MemCopy::load(devB, !b_rowmajor, block_n_base + bid * 16,
        k, N, K, LDB, taskid);
    }
    __syncthreads();
    #pragma unroll
    for (unsigned bid = 0; bid < 6; ++bid) {
      AM16K16MemCopy::store(&buffa[bid * 256], copya[bid], a_rowmajor, taskid);
      AM16K16MemCopy::store(&buffa[(6+bid) * 256], copya[bid+6], a_rowmajor, taskid);
      AM16K16MemCopy::store(&buffb[bid * 256], copyb[bid], !b_rowmajor, taskid);
    }
    __syncthreads();
    if (calc_valid) {
      acc_m48n48.accK16(&buffa[calc_m_block_id*768], &buffb[calc_n_block_id*768]);
    }
  }

  if (calc_valid) {
    std::size_t m_remain = M - warp_calc_m_base;
    std::size_t n_remain = N - warp_calc_n_base;
    acc_m48n48.storeAdd(devC + warp_calc_m_base * LDC + warp_calc_n_base,
      m_remain >= 48u ? 48u : m_remain, n_remain >= 48u ? 48u : n_remain, LDC);
  }
}

void dgemm_async(cudaStream_t &stream, const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) noexcept {

  // matrix C is row_major
  cudaFuncSetSharedMemConfig(&dgemm_kernel, cudaSharedMemBankSizeEightByte);
  if (cudaFuncSetCacheConfig(&dgemm_kernel, cudaFuncCachePreferShared) != cudaSuccess) {
    throw std::runtime_error("fail to enlarge shared memory usage for kernel");
  }

  const dim3 grid_size((N + 95u) / 96u * ((M + 191u) / 192u), 1, 1);
  const dim3 block_size(32, 2, 4);

  dgemm_kernel<<<grid_size, block_size, 0, stream>>>(
    devA, devB, devC, M, N, K,
    a_rowmajor, b_rowmajor, LDA, LDB, LDC);
}
