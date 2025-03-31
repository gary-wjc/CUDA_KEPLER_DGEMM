#include <cstddef>
#include <cassert>

struct GLoadStride {
  const std::size_t m_mIncBytes, m_kIncBytes;
  __device__ GLoadStride(std::size_t LDA, bool a_rowmajor):
    m_mIncBytes((a_rowmajor ? LDA : 1ul) << 7),
    m_kIncBytes((a_rowmajor ? 1ul : LDA) << 7) {}
};

struct GLoadPos {
  unsigned m_mPosMod16, m_kPosMod16;
  __device__ GLoadPos(unsigned warp_id, bool a_rowmajor) {
    assert(blockDim.x == 32 && warp_id < 8);
    const unsigned off1 = threadIdx.x & 15;
    const unsigned off2 = (threadIdx.x >> 4) | (warp_id << 1);
    m_mPosMod16 = a_rowmajor ? off2 : off1;
    m_kPosMod16 = a_rowmajor ? off1 : off2;
  }
};

template <unsigned MDiv16>
class GLoadState {
  unsigned m_mTasks = 0;
  std::size_t m_kTasks = 0;
  const char *m_gptr;
  const GLoadStride &m_stride;
public:
  __device__ GLoadState(std::size_t M, std::size_t K, bool a_rowmajor,
    std::size_t ctb_m_start, unsigned warp_id,
    const double *A, const GLoadStride &astride): m_stride(astride) {

    GLoadPos first_pos(warp_id, a_rowmajor);
    std::size_t mstart = first_pos.m_mPosMod16 + ctb_m_start;
    if (mstart < M) {
      if (mstart + MDiv16 * 16u <= M) m_mTasks = MDiv16;
      else m_mTasks = (15u + unsigned(M - mstart)) >> 4;
    }
    if (first_pos.m_kPosMod16 < K) {
      m_kTasks = (K - first_pos.m_kPosMod16 + 15u) >> 4;
    }
    m_gptr = reinterpret_cast<const char*>(A) + mstart * (astride.m_mIncBytes >> 4)
      + first_pos.m_kPosMod16 * (astride.m_kIncBytes >> 4);
  }
  __device__ void load_k16(double (&dst)[MDiv16]) {
    if (m_kTasks) {
      m_kTasks--;
      auto gptr = m_gptr;
      #pragma unroll
      for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
        if (md16 >= m_mTasks) dst[md16] = 0.0;
        else {
          dst[md16] = __ldg(reinterpret_cast<const double*>(gptr));
	}
        asm ("add.s64 %0,%0,%1;":"+l"(gptr):"l"(m_stride.m_mIncBytes));
      }
      asm ("add.s64 %0,%0,%1;":"+l"(m_gptr):"l"(m_stride.m_kIncBytes));
    } else {
      #pragma unroll
      for (auto &v : dst) v = 0;
    }
  }
};

__device__ __noinline__
constexpr static unsigned getSharedStartOff(unsigned m_in_ctb) {
  return ((m_in_ctb & 0xFFF0) << 4) | (m_in_ctb & 0xF);
}

__device__
constexpr static unsigned updateSharedOff(unsigned off_at_k0, unsigned kmod16) {
  return (off_at_k0 ^ kmod16) + (kmod16 << 4);
}

template <unsigned MDiv16>
class StoreShared {
  unsigned m_firstIdx;
public:
  __device__ StoreShared(unsigned warp_id, bool a_rowmajor) {
    GLoadPos firstpos(warp_id, a_rowmajor);
    m_firstIdx = updateSharedOff(firstpos.m_mPosMod16, firstpos.m_kPosMod16);
  }
  __device__ void store_k16(const double (&src)[MDiv16], double *shared_a_start) {
    double *first_dst = shared_a_start + m_firstIdx;
    #pragma unroll
    for (unsigned md16 = 0; md16 < MDiv16; ++md16) {
      first_dst[md16 << 8] = src[md16];
    }
  }
};

class WarpAccM32N32 {
  double m_acc[4][8];
  unsigned m_firstAoff, m_firstBoff;
public:
  __device__ WarpAccM32N32(
    unsigned mstart_warp_in_ctb, unsigned nstart_warp_in_ctb): m_acc{},
    m_firstAoff(getSharedStartOff(((threadIdx.x & 0xC) << 1) + mstart_warp_in_ctb)),
    m_firstBoff(getSharedStartOff((((threadIdx.x & 3) << 2) | (threadIdx.x & 16))
      + nstart_warp_in_ctb)) {

    assert(blockDim.x == 32);
  }
  __device__ void acc_k16(
    const double *shared_a_start, const double *shared_b_start) {

    auto acck4 = [&](const double *shastart, const double *shbstart,
      unsigned kabase) -> void {
      #pragma unroll
      for (unsigned k = 0; k < 4; ++k) {
        double al[8], bl[4];
        #pragma unroll
        for (unsigned n = 0; n < 4; ++n) {
          bl[n] = shbstart[updateSharedOff(n, k)];
        }
        #pragma unroll
        for (unsigned m = 0; m < 8; ++m) {
          al[m] = shastart[updateSharedOff(m, k + kabase)];
        }
        #pragma unroll
        for (unsigned n = 0; n < 4; ++n) {
          #pragma unroll
          for (unsigned m = 0; m < 8; ++m) m_acc[n][m] += bl[n] * al[m];
        }
      }
    };
    acck4(shared_a_start + m_firstAoff, shared_b_start + m_firstBoff, 0);
    acck4(shared_a_start + m_firstAoff, shared_b_start +
      updateSharedOff(m_firstBoff, 4), 4);
    unsigned k8aoff = updateSharedOff(m_firstAoff, 8);
    acck4(shared_a_start + k8aoff, shared_b_start +
      updateSharedOff(m_firstBoff, 8), 0);
    acck4(shared_a_start + k8aoff, shared_b_start +
      updateSharedOff(m_firstBoff, 12), 4);
  }
  __device__ void store(double *C, std::size_t M, std::size_t N,
    std::size_t LDC, std::size_t mstart_warp, std::size_t nstart_warp) {

    #pragma unroll
    for (unsigned n = 1; n < 4; ++n) {
      unsigned shfidx = (threadIdx.x & 16) | ((threadIdx.x + n * 4) & 0xF);
      #pragma unroll
      for (unsigned m = 0; m < 8; ++m) {
        m_acc[n][m] = __shfl_sync(0xFFFFFFFF, m_acc[n][m], shfidx);
      }
    }
    if (threadIdx.x & 8) {
      #pragma unroll
      for (unsigned m = 0; m < 8; ++m) {
        double t1 = m_acc[0][m];
        double t2 = m_acc[1][m];
        m_acc[0][m] = m_acc[2][m];
        m_acc[1][m] = m_acc[3][m];
        m_acc[2][m] = t1;
        m_acc[3][m] = t2;
      }
    }
    if (threadIdx.x & 4) {
      #pragma unroll
      for (unsigned m = 0; m < 8; ++m) {
        double t1 = m_acc[0][m];
        m_acc[0][m] = m_acc[3][m];
        m_acc[3][m] = m_acc[2][m];
        m_acc[2][m] = m_acc[1][m];
        m_acc[1][m] = t1;
      }
    }
    double *warp_c = C + mstart_warp * LDC + nstart_warp;
    unsigned mleft = 0;
    if (mstart_warp < M) {
      if (mstart_warp + 32u < M) mleft = 32;
      else mleft = M - mstart_warp;
    }
    const unsigned baseidx = ((threadIdx.x & 3) << 2) | (threadIdx.x & 16);
    #pragma unroll
    for (unsigned n = 0; n < 4; ++n) {
      unsigned npos = baseidx + ((8 + n - (threadIdx.x >> 2)) & 3);
      if (npos + nstart_warp < N) {
        #pragma unroll
        for (unsigned m = 0; m < 8; ++m) {
          if (!mleft) return;
          mleft--;
          warp_c[npos] += m_acc[n][m];
          warp_c += LDC;
        }
      } else {
        if (mleft <= 8) return;
        mleft -= 8;
        warp_c += LDC * 8u;
      }
    }
  }
};

__launch_bounds__(512, 1) __global__
void dgemm_kernel(const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) {

  assert(blockDim.y == 4 && blockDim.z == 4);
  const std::size_t nblks = (N+127u) >> 7;
  const std::size_t mblkid = blockIdx.x / nblks;
  const std::size_t nblkid_off = blockIdx.x % nblks;
  const std::size_t nblkid = mblkid & 1u ?
    (nblks - 1u - nblkid_off) : nblkid_off;
  const std::size_t ctb_m_base = mblkid * 128u;
  const std::size_t ctb_n_base = nblkid * 128u;
  if (ctb_m_base >= M || ctb_n_base >= N) return;

  const unsigned warp_id = threadIdx.z * blockDim.y + threadIdx.y;
  const bool gld_krowmajor = warp_id >= 8 ? !b_rowmajor : a_rowmajor;
  GLoadStride stride(warp_id >= 8 ? LDB : LDA, gld_krowmajor);
  GLoadState<8> global_load_state(warp_id >= 8 ? N : M, K,
    gld_krowmajor,
    warp_id >= 8 ? ctb_n_base : ctb_m_base, warp_id & 7,
    warp_id >= 8 ? devB : devA, stride);
  __shared__ double buffa[16*16*8], buffb[16*16*8];
  StoreShared<8> store_shared(warp_id & 7, gld_krowmajor);

  unsigned warp_mstart_in_ctb = threadIdx.z << 5;
  unsigned warp_nstart_in_ctb = threadIdx.y << 5;
  bool consume = (warp_mstart_in_ctb + ctb_m_base < M) &&
    (warp_nstart_in_ctb + ctb_n_base < N);
  WarpAccM32N32 wacc(warp_mstart_in_ctb, warp_nstart_in_ctb);

  for (std::size_t k = 0; k < K; k += 16) {
    double ld[8];
    global_load_state.load_k16(ld);
    __syncthreads();
    store_shared.store_k16(ld, warp_id >= 8 ? buffb : buffa);
    __syncthreads();
    if (consume) wacc.acc_k16(buffa, buffb);
  }
  if (consume) wacc.store(devC, M, N, LDC, warp_mstart_in_ctb + ctb_m_base,
    warp_nstart_in_ctb + ctb_n_base);
}

void dgemm_async(cudaStream_t &stream,
  const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) noexcept {

  const dim3 grid_size((N + 127u) / 128u * ((M + 127u) / 128u), 1, 1);
  const dim3 block_size(32, 4, 4);

  dgemm_kernel<<<grid_size, block_size, 0, stream>>>(
    devA, devB, devC, M, N, K,
    a_rowmajor, b_rowmajor,
    LDA, LDB, LDC);
}
