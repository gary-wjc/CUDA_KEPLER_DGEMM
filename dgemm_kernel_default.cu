//nvcc -arch=sm_xx --shared -DNDEBUG --compiler-options -fPIC,-O3 -Xcicc -O3 -keep -Xptxas -v uuu.cu -o libdgemm_kernel.so
#if __CUDA_ARCH__ >= 800
#define DGEMM_SM80
#endif

#include <cassert>
#include <cstddef>

struct GLoadFirstPos {
  const unsigned m_mMod16, m_kMod16;
  __device__ GLoadFirstPos(unsigned warp_id, bool row_major):
    m_mMod16(row_major ? ((threadIdx.x & 1) | (warp_id << 1)) : (threadIdx.x & 15)),
    m_kMod16(row_major ? ((threadIdx.x & 14) | (threadIdx.x >> 4)) :
      ((warp_id << 1) | (threadIdx.x >> 4))) {

    assert(blockDim.x == 32 && warp_id < 8);
  }
};

class MemTransfer {
  unsigned m_sharedFirstPos;
  const char *m_gPtr;
  std::size_t m_kStrideBytes;
  std::size_t m_mnStrideBytes;
  unsigned m_mnTasks;
  std::size_t m_kTasks;
public:
  __device__ MemTransfer(const double *A, const double *B,
    std::size_t M, std::size_t N, std::size_t K, std::size_t LDA,
    std::size_t LDB, bool a_rowmajor, bool b_rowmajor,
    std::size_t block_start_m, std::size_t block_start_n) {
    assert(blockDim.y * blockDim.z == 16);
    const int warp_id = threadIdx.z * blockDim.y + threadIdx.y;
    const bool select_b = warp_id & 8;
    const bool kmajor = select_b ? !b_rowmajor : a_rowmajor;
    const GLoadFirstPos first_pos(warp_id & 7, kmajor);
    const unsigned &koff = first_pos.m_kMod16;
    const unsigned &mnoff = first_pos.m_mMod16;
    m_sharedFirstPos = koff == 15 ? ((mnoff >> 1) * 34u | (mnoff & 1)) :
      ((koff >> 1) * 34u + 2u + (mnoff | ((koff & 1) << 4)));
    std::size_t mn_max = select_b ? N : M;
    std::size_t mn_first = (select_b ? block_start_n : block_start_m) + mnoff;
    std::size_t LD = select_b ? LDB : LDA;
    m_kStrideBytes = (kmajor ? 1ul : LD) << 7;
    m_mnStrideBytes = (kmajor ? LD : 1ul) << 7;
    m_gPtr = reinterpret_cast<const char*>(select_b ? B : A) +
      (m_mnStrideBytes >> 4) * mn_first + (m_kStrideBytes >> 4) * koff;
    auto get_tasks = [](std::size_t start, std::size_t end)->std::size_t {
      if (end <= start) return 0;
      return (end + 15u - start) >> 4;
    };
    std::size_t mn_tasks = get_tasks(mn_first, mn_max);
    m_mnTasks = mn_tasks > 8 ? 8u : mn_tasks;
    m_kTasks = get_tasks(koff, K);
  }
#ifdef DGEMM_SM80
#pragma message "enable cp.async"
  __device__ void issue_transfer(double2 *shared) {
    unsigned tbytes = 0;
    if (m_kTasks) {
      tbytes = 8;
      m_kTasks--;
    }
    auto gptr = m_gPtr;
    asm ("cvta.to.global.u64 %0,%0;":"+l"(gptr));
    unsigned baseoff = (threadIdx.z * blockDim.y + threadIdx.y & 8) << 8;
    double *begin = (&shared[0].x) + baseoff + m_sharedFirstPos;
    asm ("cvta.to.shared.u64 %0,%0;":"+l"(begin));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j >= m_mnTasks) break;
      asm volatile ("cp.async.ca.shared.global [%0],[%1],8,%2;"
        ::"l"(begin),"l"(gptr),"r"(tbytes));
      gptr += m_mnStrideBytes;
      begin += 256;
    }
    m_gPtr += m_kStrideBytes;
    asm volatile ("cp.async.commit_group;");
  }
#endif
  __device__ void load(double (&recv)[8]) {
    if (m_kTasks) {
      m_kTasks--;
      auto gptr = m_gPtr;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j >= m_mnTasks) break;
        recv[j] = __ldg((const double*)gptr);
        gptr += m_mnStrideBytes;
      }
      m_gPtr += m_kStrideBytes;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j >= m_mnTasks) break;
        recv[j] = 0;
      }
    }
  }
  __device__ void store(const double (&recv)[8], double2 *shared) const {
    unsigned baseoff = (threadIdx.z * blockDim.y + threadIdx.y & 8) << 8;
    double *begin = (&shared[0].x) + baseoff + m_sharedFirstPos;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < m_mnTasks) begin[j << 8] = recv[j];
    }
  }
};

class WarpAccumulator {
  double2 m_acc[2][2][2][2];
  bool m_calcValid;
public:
  __device__ WarpAccumulator(std::size_t M, std::size_t N,
    std::size_t block_m_base, std::size_t block_n_base): m_acc{} {

    assert(blockDim.y == 4 && blockDim.z == 4 && blockDim.x == 32);
    std::size_t npos_start = block_n_base + (threadIdx.y << 5);
    std::size_t mpos_start = block_m_base + (threadIdx.z << 5);
    m_calcValid = npos_start < N && mpos_start < M;
  }
  __device__ void acc_k16(const double2 *shared) {
    if (!m_calcValid) return;
#ifdef DGEMM_SM80
#pragma message "use tensor core"
    unsigned mnstartdiv2 = threadIdx.x >> 2;
    unsigned kstartdiv2 = (threadIdx.x & 3) << 1;
    unsigned offins32 = mnstartdiv2 + 1 + kstartdiv2 * 17u;
    const double2 *abase = &shared[threadIdx.z * 256u];
    const double2 *bbase = &shared[(threadIdx.y+4) * 256u];
    const double2 *astart = &abase[offins32];
    const double2 *bstart = &bbase[offins32];
    auto acc_k4 = [&]()->void {
      double2 a[2] = { *astart, astart[128] };
      double2 b[2] = { *bstart, bstart[128] };
#pragma unroll
      for (short nseg = 0; nseg < 2; ++nseg) {
        const double2 &bv = b[nseg];
#pragma unroll
        for (short mseg = 0; mseg < 2; ++mseg) {
          const double2 &av = a[mseg];
          asm (
#if __CUDA_ARCH__ >= 900
#pragma message "optimize for Hopper arch"
            "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64"
	      " {%0,%1,%4,%5},{%8,%9},{%10},{%0,%1,%4,%5};"
            "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64"
	      " {%2,%3,%6,%7},{%8,%9},{%11},{%2,%3,%6,%7};"
#else
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1},{%8},{%10},{%0,%1};"
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%2,%3},{%8},{%11},{%2,%3};"
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%4,%5},{%9},{%10},{%4,%5};"
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%6,%7},{%9},{%11},{%6,%7};"
#endif
           :"+d"(m_acc[nseg][mseg][0][0].x),"+d"(m_acc[nseg][mseg][0][1].x),
            "+d"(m_acc[nseg][mseg][0][0].y),"+d"(m_acc[nseg][mseg][0][1].y),
            "+d"(m_acc[nseg][mseg][1][0].x),"+d"(m_acc[nseg][mseg][1][1].x),
            "+d"(m_acc[nseg][mseg][1][0].y),"+d"(m_acc[nseg][mseg][1][1].y)
           :"d"(bv.x),"d"(bv.y),"d"(av.x),"d"(av.y));
        }
      }
    };
    acc_k4();
    astart += 8u;
    bstart += 8u;
    acc_k4();
    astart += 9u;
    bstart += 9u;
    acc_k4();
    if (kstartdiv2 == 6) offins32 = mnstartdiv2 * 17u;
    else offins32 += 25u;
    astart = &abase[offins32];
    bstart = &bbase[offins32];
    acc_k4();
#else
    unsigned mstartdiv2 = (threadIdx.x & 3) << 1;
    unsigned nstartdiv2 = threadIdx.x >> 2;
    const double2 *astart = &shared[threadIdx.z * 256u + mstartdiv2 + 1];
    const double2 *bstart = &shared[(threadIdx.y+4) * 256u + nstartdiv2 + 1];

    auto acc_k1 = [&](unsigned short a2sep)->void {
      double2 a[2][2];
      a[0][0] = *astart;
      a[0][1] = astart[a2sep];
      a[1][0] = astart[128];
      a[1][1] = astart[128+a2sep];
      double2 b[2];
      b[0] = *bstart;
      b[1] = bstart[128];
#pragma unroll
      for (short nseg = 0; nseg < 2; ++nseg) {
        const double2 &b1 = b[nseg];
#pragma unroll
	for (short mseg = 0; mseg < 2; ++mseg) {
          const double2 &a1 = a[mseg][0];
          const double2 &a2 = a[mseg][1];

          m_acc[nseg][mseg][0][0].x += a1.x * b1.x; //m0
          m_acc[nseg][mseg][0][0].y += a1.y * b1.x; //m1
          m_acc[nseg][mseg][0][1].x += a2.x * b1.x; //m2
          m_acc[nseg][mseg][0][1].y += a2.y * b1.x; //m3

          m_acc[nseg][mseg][1][0].x += a1.x * b1.y;
          m_acc[nseg][mseg][1][0].y += a1.y * b1.y;
          m_acc[nseg][mseg][1][1].x += a2.x * b1.y;
          m_acc[nseg][mseg][1][1].y += a2.y * b1.y;
        }
      }
    };

#pragma unroll
    for (int kd2 = 0; kd2 < 7; ++kd2) {
      acc_k1(1);
      astart += 8;
      bstart += 8;
      acc_k1(1);
      astart += 9;
      bstart += 9;
    }
    acc_k1(1);
    astart = &shared[threadIdx.z * 256u + mstartdiv2 * 17];
    bstart = &shared[(threadIdx.y+4) * 256u + nstartdiv2 * 17];
    acc_k1(17);
#endif
  }
  __device__ void store(double *C, std::size_t M, std::size_t N, std::size_t LDC,
    std::size_t block_m_base, std::size_t block_n_base) {

    if (!m_calcValid) return;
    std::size_t npos_start = block_n_base + (threadIdx.y << 5) +
      (threadIdx.x & 1) + (threadIdx.x >> 2 << 1);
    std::size_t mpos_start = block_m_base + (threadIdx.z << 5) +
      ((threadIdx.x & 2) << 2);
    #pragma unroll
    for (short nseg = 0; nseg < 2; ++nseg) {
      #pragma unroll
      for (short mseg = 0; mseg < 2; ++mseg) {
        auto shfl = [](double2 var, int idx)->double2 {
          return make_double2(__shfl_sync(0xFFFFFFFF, var.x, idx),
            __shfl_sync(0xFFFFFFFF, var.y, idx));
        };
        m_acc[nseg][mseg][1][0] = shfl(m_acc[nseg][mseg][1][0], threadIdx.x ^ 1);
        m_acc[nseg][mseg][1][1] = shfl(m_acc[nseg][mseg][1][1], threadIdx.x ^ 1);
        if (threadIdx.x & 1) {
          double2 t0 = m_acc[nseg][mseg][1][0];
          double2 t1 = m_acc[nseg][mseg][1][1];
          m_acc[nseg][mseg][1][0] = m_acc[nseg][mseg][0][0];
          m_acc[nseg][mseg][1][1] = m_acc[nseg][mseg][0][1];
          m_acc[nseg][mseg][0][0] = t0;
          m_acc[nseg][mseg][0][1] = t1;
        }
        std::size_t mpos = mpos_start + mseg * 16u;
        std::size_t npos = npos_start + nseg * 16u;
        auto store2 = [&](double2 val)->void {
          if (npos < N) {
            if (mpos < M) C[mpos * LDC + npos] += val.x;
            if (mpos + 1u < M) C[(mpos+1u) * LDC + npos] += val.y;
          }
        };
	store2(m_acc[nseg][mseg][0][0]);
        mpos += 2;
        store2(m_acc[nseg][mseg][0][1]);
        mpos += 2;
        npos ^= 1;
        store2(m_acc[nseg][mseg][1][0]);
        mpos += 2;
        store2(m_acc[nseg][mseg][1][1]);
      }
    }
  }
};

__launch_bounds__(512, 1) __global__ void dgemm_kernel(
  const double *A, const double *B, double *C,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) {

  assert(blockDim.y == 4 && blockDim.z == 4);
  unsigned nblkid = blockIdx.y;
  unsigned mblkid = blockIdx.z;
  const unsigned nblks = (N+127u) >> 7;
  const unsigned mblks = (M+127u) >> 7;
  if (nblkid < (nblks & unsigned(-4)) &&
    mblkid < (mblks & unsigned(-4))) {
    //perform reordered mapping of (4, nblks&-4) to increase L2-cache hit rate
    unsigned short submid = nblkid & 3u;
    if (nblkid & 4u) submid = 3u - submid;
    unsigned new_mblkid = (mblkid & unsigned(-4)) | submid;
    nblkid = (nblkid >> 2) + (nblks >> 2) * (mblkid & 3u);
    mblkid = new_mblkid;
  }
  const std::size_t block_m_base = std::size_t(mblkid) << 7;
  const std::size_t block_n_base = std::size_t(nblkid) << 7;
  if (block_m_base >= M || block_n_base >= N) return;
  MemTransfer mtr(A, B, M, N, K, LDA, LDB, a_rowmajor, b_rowmajor,
    block_m_base, block_n_base);
  WarpAccumulator acc(M, N, block_m_base, block_n_base);

  extern __shared__ double2 shared[]; //double2 shared[2048];
#ifdef DGEMM_SM80
#pragma message "use double-buffered shared memory"
  __shared__ double2 shared2[2048];
  mtr.issue_transfer(shared);
  for (std::size_t k = 0; k < K; k += 16) {
    mtr.issue_transfer(k & 16u ? shared : shared2);
    asm volatile ("cp.async.wait_group 1;");
    __syncthreads();
    acc.acc_k16(k & 16u ? shared2 : shared);
  }
#else
  double gload[8];
  for (std::size_t k = 0; k < K; k += 16) {
    mtr.load(gload);
    __syncthreads();
    mtr.store(gload, shared);
    __syncthreads();
    acc.acc_k16(shared);
  }
#endif
  acc.store(C, M, N, LDC, block_m_base, block_n_base);
}

void dgemm_async(cudaStream_t &stream, const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) noexcept {

  unsigned shared_bytes = 32768;
  cudaFuncSetAttribute(&dgemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
    shared_bytes);
  cudaFuncSetSharedMemConfig(&dgemm_kernel, cudaSharedMemBankSizeEightByte);

  //TODO: when gridDim.x or gridDim.y exceed device capability, call dgemm_kernel in loops
  const dim3 grid_size(1, (N+127u) >> 7, (M+127u) >> 7);
  const dim3 block_size(32, 4, 4);

  dgemm_kernel<<<grid_size, block_size, shared_bytes, stream>>>(
    devA, devB, devC, M, N, K,
    a_rowmajor, b_rowmajor, LDA, LDB, LDC);
}
