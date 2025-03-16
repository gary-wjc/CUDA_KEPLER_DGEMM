#include <cassert>
#include <cstdlib>

namespace cuda_thread {

//shared memory pattern inspired from the description of WGMMA in NVIDIA PTX manual 8.7
class ShmLayout {
protected:
  __device__ constexpr
  static unsigned getSharedIdx(unsigned subm, unsigned subk) noexcept {
    return (subk << 5) | (subm ^ subk);
  }
};

template <unsigned MDiv32>
class MemTransferA: protected ShmLayout {
  //can also be used for matrix B: A_m32=>B_n32, subM_end=>subN_end, LDA=>LDB, a_rowmajor=>b_colmajor
  //m_subM, m_subK, m_sharedMem only for first M32
  const unsigned m_subM, m_subK;
  const unsigned m_sharedMemOff;
  const std::ptrdiff_t m_m32Stride;
  const std::ptrdiff_t m_globalMemStride;

  const char *m_globalMem;
  unsigned m_commonTasks;
  unsigned m_tasks[MDiv32];
  double m_loaded[MDiv32];

public:
  //task_id: [0,16)
  //shared_block: double[16][32] (a block of shared memory, holding m32k16 submat)
  //each warp manipulate 1/16 of shared_block, that is 32 double numbers
  //each warp is assigned a task_id
  __device__ MemTransferA(const double *A_m32,
    unsigned task_id, unsigned block_mlen, std::size_t K,
    std::size_t LDA, bool a_rowmajor) noexcept:
    m_subM(a_rowmajor ? (task_id + (threadIdx.x >> 4 << 4)) : threadIdx.x),
    m_subK(a_rowmajor ? (threadIdx.x % 16) : task_id),
    m_sharedMemOff(getSharedIdx(m_subM, m_subK)),
    m_m32Stride(a_rowmajor ? 256u * LDA : 256u),
    m_globalMemStride(std::ptrdiff_t(a_rowmajor ? 128u : 128u * LDA)
      - m_m32Stride * MDiv32),
    m_globalMem(reinterpret_cast<const char*>(A_m32
      + m_subK * (a_rowmajor ? 1u : LDA)
      + m_subM * (a_rowmajor ? LDA : 1u))) {

    assert(warpSize == 32 && blockDim.x == 32);
    assert(K <= (1LL << (4 + sizeof(unsigned) * CHAR_BIT)));
    unsigned max_m = (a_rowmajor ? (task_id + 16) : 31) + (MDiv32 - 1u) * 32u;
    if (max_m >= block_mlen) m_commonTasks = 0;
    else {
      unsigned max_k = a_rowmajor ? 15 : task_id;
      if (max_k >= K) m_commonTasks = 0;
      else m_commonTasks = (K - max_k + 15u) >> 4;
    }
    for (unsigned i = 0; i < MDiv32; ++i) {
      if (block_mlen <= 32u * i + m_subM) m_tasks[i] = 0;
      else if (m_subK >= K) m_tasks[i] = 0;
      else m_tasks[i] = (K - m_subK + 15u) >> 4;
      m_tasks[i] -= m_commonTasks;
    }
  }

  __device__ void load() noexcept {
    if (m_commonTasks) {
      m_commonTasks--;
      #pragma unroll
      for (unsigned i = 0; i < MDiv32; ++i) {
        m_loaded[i] = __ldg(reinterpret_cast<const double*>(m_globalMem));
        m_globalMem += m_m32Stride;
      }
      m_globalMem += m_globalMemStride;
      return;
    }
    #pragma unroll
    for (unsigned i = 0; i < MDiv32; ++i) {
      unsigned &tasks = m_tasks[i];
      double &loaded = m_loaded[i];
      if (tasks) {
        tasks--;
        loaded = __ldg(reinterpret_cast<const double*>(m_globalMem));
      } else loaded = 0;
      m_globalMem += m_m32Stride;
    }
    m_globalMem += m_globalMemStride;
  }

  __device__ void store(double *shared_mem) const noexcept {
    shared_mem += m_sharedMemOff;
    #pragma unroll
    for (unsigned i = 0; i < MDiv32; ++i) {
      shared_mem[i * 512u] = m_loaded[i];
    }
  }
}; // class MemTransferA

#if __CUDA_ARCH__ >= 800
#pragma message "enable tensor-core fp64 acceleration"
#define DGEMM_USE_TENSOR_CORE
#endif

class Accumulator: protected ShmLayout {
  double c00, c01, c02, c03;
  double c10, c11, c12, c13;
  double c20, c21, c22, c23;
  double c30, c31, c32, c33;
  double c40, c41, c42, c43;
  double c50, c51, c52, c53;
  double c60, c61, c62, c63;
  double c70, c71, c72, c73;
#ifdef DGEMM_USE_TENSOR_CORE
  const short m_firstIndex;
#else
  const short m_startM1, m_startM2, m_startN;
#endif
  __device__ static double shfl(double var, unsigned src_idx) noexcept {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#pragma message "use shfl_sync"
    return __shfl_sync(0xFFFFFFFF, var, src_idx);
#else
    return __shfl(var, src_idx);
#endif
  }
#ifdef DGEMM_USE_TENSOR_CORE
  __device__ static void mmaM8N8K4(double &c1, double &c2,
    double a, double b) noexcept {

    asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
      "{%0,%1},{%2},{%3},{%0,%1};":"+d"(c1),"+d"(c2):"d"(a),"d"(b),"d"(c1),"d"(c2));
  }
  __noinline__
  __device__ static short getFirstIndex(int tidx) noexcept {
    int k = (tidx & 3) << 2;
    int m = tidx >> 2 << 2;
    return getSharedIdx(m, k);
  }
#endif
public:
  __device__ Accumulator() noexcept:
    c00(0.0), c01(0.0), c02(0.0), c03(0.0),
    c10(0.0), c11(0.0), c12(0.0), c13(0.0),
    c20(0.0), c21(0.0), c22(0.0), c23(0.0),
    c30(0.0), c31(0.0), c32(0.0), c33(0.0),
    c40(0.0), c41(0.0), c42(0.0), c43(0.0),
    c50(0.0), c51(0.0), c52(0.0), c53(0.0),
    c60(0.0), c61(0.0), c62(0.0), c63(0.0),
    c70(0.0), c71(0.0), c72(0.0), c73(0.0),
#ifdef DGEMM_USE_TENSOR_CORE
    m_firstIndex(getFirstIndex(threadIdx.x))
#else
    m_startM1(threadIdx.x % 4u * 8u), m_startM2(m_startM1 + 4u),
    m_startN(threadIdx.x / 4u * 4u)
#endif
  {

    assert(warpSize == 32 && blockDim.x == 32);
  }

  __device__ void accK16(const double *shared_blk_a, const double *shared_blk_b) noexcept {
#ifdef DGEMM_USE_TENSOR_CORE
    auto acck4 = [&](double a0, double a1, double a2, double a3,
      double b0, double b1, double b2, double b3)->void {

      mmaM8N8K4(c00, c40, b0, a0);
      mmaM8N8K4(c01, c41, b1, a0);
      mmaM8N8K4(c02, c42, b2, a0);
      mmaM8N8K4(c03, c43, b3, a0);
      mmaM8N8K4(c10, c50, b0, a1);
      mmaM8N8K4(c11, c51, b1, a1);
      mmaM8N8K4(c12, c52, b2, a1);
      mmaM8N8K4(c13, c53, b3, a1);
      mmaM8N8K4(c20, c60, b0, a2);
      mmaM8N8K4(c21, c61, b1, a2);
      mmaM8N8K4(c22, c62, b2, a2);
      mmaM8N8K4(c23, c63, b3, a2);
      mmaM8N8K4(c30, c70, b0, a3);
      mmaM8N8K4(c31, c71, b1, a3);
      mmaM8N8K4(c32, c72, b2, a3);
      mmaM8N8K4(c33, c73, b3, a3);
    };
    const double *ap = shared_blk_a + m_firstIndex;
    const double *bp = shared_blk_b + m_firstIndex;
    #pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
      acck4(ap[getSharedIdx(0, k)], ap[getSharedIdx(1, k)], ap[getSharedIdx(2, k)],
        ap[getSharedIdx(3, k)], bp[getSharedIdx(0, k)], bp[getSharedIdx(1, k)],
        bp[getSharedIdx(2, k)], bp[getSharedIdx(3, k)]);
    }
#else
    auto acck1 = [&](double a0, double a1, double a2, double a3,
      double a4, double a5, double a6, double a7,
      double b0, double b1, double b2, double b3)->void {

      auto acc4 = [&](double &c1, double &c2,
        double &c3, double &c4, double a)->void {
        c1 += a * b0;
        c2 += a * b1;
        c3 += a * b2;
        c4 += a * b3;
      };
      acc4(c00, c01, c02, c03, a0);
      acc4(c10, c11, c12, c13, a1);
      acc4(c20, c21, c22, c23, a2);
      acc4(c30, c31, c32, c33, a3);
      acc4(c40, c41, c42, c43, a4);
      acc4(c50, c51, c52, c53, a5);
      acc4(c60, c61, c62, c63, a6);
      acc4(c70, c71, c72, c73, a7);
    };
    auto acck4 = [&](unsigned short kstart)->void {
      const double *ap1 = shared_blk_a + getSharedIdx(m_startM1, kstart);
      const double *ap2 = shared_blk_a + getSharedIdx(m_startM2, kstart);
      const double *bp = shared_blk_b + getSharedIdx(m_startN, kstart);

      #pragma unroll
      for (unsigned k = 0; k < 4; ++k) {
        acck1(ap1[getSharedIdx(0, k)], ap1[getSharedIdx(1, k)],
          ap1[getSharedIdx(2, k)], ap1[getSharedIdx(3, k)],
          ap2[getSharedIdx(0, k)], ap2[getSharedIdx(1, k)], 
          ap2[getSharedIdx(2, k)], ap2[getSharedIdx(3, k)],
          bp[getSharedIdx(0, k)], bp[getSharedIdx(1, k)],
          bp[getSharedIdx(2, k)], bp[getSharedIdx(3, k)]);
      }
    };
    #pragma unroll 
    for (unsigned short k = 0; k < 16; k += 4) acck4(k);
#endif
  }

  __device__ void storeC(double *c_m32n32, unsigned LDC,
    unsigned m, unsigned n) noexcept {
    //first, exchange data between threads
    auto shuf8 = [](double &c1, double &c2, double &c3, double &c4,
      double &c5, double &c6, double &c7, double &c8, unsigned src_idx)->void {
      c1 = shfl(c1, src_idx);
      c2 = shfl(c2, src_idx);
      c3 = shfl(c3, src_idx);
      c4 = shfl(c4, src_idx);
      c5 = shfl(c5, src_idx);
      c6 = shfl(c6, src_idx);
      c7 = shfl(c7, src_idx);
      c8 = shfl(c8, src_idx);
    };
    const unsigned baseidx = threadIdx.x / 4u * 4u;
    shuf8(c01, c11, c21, c31, c41, c51, c61, c71,
      baseidx + (threadIdx.x + 1u) % 4u);
    shuf8(c02, c12, c22, c32, c42, c52, c62, c72,
      baseidx + (threadIdx.x + 2u) % 4u);
    shuf8(c03, c13, c23, c33, c43, c53, c63, c73,
      baseidx + (threadIdx.x + 3u) % 4u);
    if (threadIdx.x % 4u >= 2u) {
      auto move2 = [](double &c0, double &c1, double &c2, double &c3)->void {
        double t0 = c0, t1 = c1;
        c0 = c2;
        c1 = c3;
        c2 = t0;
        c3 = t1;
      };
      move2(c00, c01, c02, c03);
      move2(c10, c11, c12, c13);
      move2(c20, c21, c22, c23);
      move2(c30, c31, c32, c33);
      move2(c40, c41, c42, c43);
      move2(c50, c51, c52, c53);
      move2(c60, c61, c62, c63);
      move2(c70, c71, c72, c73);
    }
    if (threadIdx.x % 2u) {
      auto move1 = [](double &c0, double &c1, double &c2, double &c3)->void {
        double t0 = c0;
        c0 = c3;
        c3 = c2;
        c2 = c1;
        c1 = t0;
      };
      move1(c00, c01, c02, c03);
      move1(c10, c11, c12, c13);
      move1(c20, c21, c22, c23);
      move1(c30, c31, c32, c33);
      move1(c40, c41, c42, c43);
      move1(c50, c51, c52, c53);
      move1(c60, c61, c62, c63);
      move1(c70, c71, c72, c73);
    }
    auto write_m8n1 = [](double *c, unsigned ldc, unsigned valid_m,
      double c0, double c1, double c2, double c3,
      double c4, double c5, double c6, double c7)->void {
      auto write_1 = [&](double c1)->void {
        if (valid_m) {
          *c += c1;
          c += ldc;
          valid_m--;
        }
      };
      write_1(c0);
      write_1(c1);
      write_1(c2);
      write_1(c3);
      write_1(c4);
      write_1(c5);
      write_1(c6);
      write_1(c7);
    };
    unsigned npos = baseidx + (32 - threadIdx.x) % 4u;
    if (npos < n) write_m8n1(c_m32n32 + npos, LDC, m,
      c00, c10, c20, c30, c40, c50, c60, c70);
    if (m <= 8) return;
    c_m32n32 += LDC * 8u;
    m -= 8;
    // done first m8n32 write
    npos = baseidx + (33 - threadIdx.x) % 4u;
    if (npos < n) write_m8n1(c_m32n32 + npos, LDC, m,
      c01, c11, c21, c31, c41, c51, c61, c71);
    if (m <= 8) return;
    c_m32n32 += LDC * 8u;
    m -= 8;
    // done second m8n32 write
    npos = baseidx + (34 - threadIdx.x) % 4u;
    if (npos < n) write_m8n1(c_m32n32 + npos, LDC, m,
      c02, c12, c22, c32, c42, c52, c62, c72);
    if (m <= 8) return;
    c_m32n32 += LDC * 8u;
    m -= 8;
    // done third m8n32 write
    npos = baseidx + (35 - threadIdx.x) % 4u;
    if (npos < n) write_m8n1(c_m32n32 + npos, LDC, m,
      c03, c13, c23, c33, c43, c53, c63, c73);
  }
}; // class Accumulator

} // namespace cuda_thread

namespace impl {

__global__ void dgemm_kernel(const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) {

  assert(blockDim.y == 4 && blockDim.z == 4);
  const std::size_t nblks = (N+127u) >> 7;
  const std::size_t mblkid = blockIdx.x / nblks;
  const std::size_t nblkid_off = blockIdx.x % nblks;
  const std::size_t nblkid = mblkid & 1u ?
    (nblks - 1u - nblkid_off) : nblkid_off;
  const std::size_t block_m_base = mblkid * 128u;
  const std::size_t block_n_base = nblkid * 128u;
  if (block_m_base >= M || block_n_base >= N) return;
  const std::size_t a_m_inc = a_rowmajor ? LDA : 1;
  const std::size_t b_n_inc = b_rowmajor ? 1 : LDB;
  const double *block_a_base = devA + block_m_base * a_m_inc;
  const double *block_b_base = devB + block_n_base * b_n_inc;

  __shared__ double shared_blockA[512 * 4];
  __shared__ double shared_blockB[512 * 4];
  auto get_subm_end = [](std::size_t M, std::size_t m_base)->unsigned {
    if (M >= m_base + 32u) return 32u;
    if (M <= m_base) return 0;
    return M - m_base;
  };
  const unsigned copy_taskid = threadIdx.z * blockDim.y + threadIdx.y;
  cuda_thread::MemTransferA<4> mta1(block_a_base,
    copy_taskid, M - block_m_base, K, LDA, a_rowmajor);
  cuda_thread::MemTransferA<4> mtb1(block_b_base,
    copy_taskid, N - block_n_base, K, LDB, !b_rowmajor);

  const unsigned calc_m_block_id = threadIdx.z;
  const unsigned calc_n_block_id = threadIdx.y;
  const std::size_t warp_calc_m_base = block_m_base + calc_m_block_id * 32u;
  const std::size_t warp_calc_n_base = block_n_base + calc_n_block_id * 32u;
  const unsigned warp_calc_m = get_subm_end(M, warp_calc_m_base);
  const unsigned warp_calc_n = get_subm_end(N, warp_calc_n_base);
  double * const warp_submat_c = devC + warp_calc_n_base + warp_calc_m_base * LDC;
  const bool calc_valid = warp_calc_m && warp_calc_n;
  cuda_thread::Accumulator acc_m8n4;
  const unsigned shoffblka = calc_m_block_id * 512u;
  const unsigned shoffblkb = calc_n_block_id * 512u;

  for (std::ptrdiff_t kleft = K; kleft > 0; kleft -= 16) {
    mta1.load();
    mtb1.load();
    __syncthreads();
    mta1.store(shared_blockA);
    mtb1.store(shared_blockB);
    __syncthreads();
    if (calc_valid) {
      acc_m8n4.accK16(shared_blockA + shoffblka, shared_blockB + shoffblkb);
    }
  }

  if (calc_valid) {
    acc_m8n4.storeC(warp_submat_c, LDC, warp_calc_m, warp_calc_n);
  }
}

}

void dgemm_async(cudaStream_t &stream, const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) noexcept {

  // matrix C is row_major
  cudaFuncSetSharedMemConfig(&impl::dgemm_kernel, cudaSharedMemBankSizeEightByte);
    
  const dim3 grid_size((N + 127u) / 128u * ((M + 127u) / 128u), 1, 1);
  const dim3 block_size(32, 4, 4);

  impl::dgemm_kernel<<<grid_size, block_size, 0, stream>>>(
    devA, devB, devC, M, N, K,
    a_rowmajor, b_rowmajor, LDA, LDB, LDC);
}
