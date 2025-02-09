#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

class GPUMem {
  void *m_addr = nullptr;
  const std::size_t m_size;
public:
  GPUMem(std::size_t size_bytes): m_size(size_bytes) {
    auto code = cudaSuccess;
    if ((code = cudaMalloc(&m_addr, size_bytes)) != cudaSuccess) {
      throw std::runtime_error("fail to allocate " + std::to_string(size_bytes)
        + " bytes of GPU memory, error: " + cudaGetErrorString(code));
    }
  }
  void copyToGPU(const void *host) const {
    if (cudaMemcpy(m_addr, host, m_size, cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("fail to copy " + std::to_string(m_size)
        + " bytes of data from host to GPU");
    }
  }
  void copyToHost(void *host) const {
    if (cudaMemcpy(host, m_addr, m_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("fail to copy " + std::to_string(m_size)
        + " bytes of data from GPU to host");
    }
  }
  void *get() const noexcept { return m_addr; }
  ~GPUMem() { cudaFree(m_addr); }
};

class Stream {
  cudaStream_t m_stream;
public:
  Stream() {
    if (cudaStreamCreate(&m_stream) != cudaSuccess) {
      throw std::runtime_error("fail to create stream");
    }
  }
  ~Stream() {
    cudaStreamDestroy(m_stream);
  }
  cudaStream_t &get() noexcept { return m_stream; }
};

extern void dgemm_async(cudaStream_t &stream, const double *devA, const double *devB, double *devC,
  std::size_t M, std::size_t N, std::size_t K, bool a_rowmajor, bool b_rowmajor,
  std::size_t LDA, std::size_t LDB, std::size_t LDC) noexcept;

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <pad>\n";
    return 0;
  }
  const std::size_t M = std::atol(argv[1]);
  const std::size_t N = std::atol(argv[2]);
  const std::size_t K = std::atol(argv[3]);
  const std::size_t pad = std::atol(argv[4]);
  std::cout << "M = " << M << ", N = " << N << ", K = " << K
    << ", pad = " << pad << std::endl;

  std::vector<double> A((M + pad) * (K + pad));
  std::vector<double> B((N + pad) * (K + pad));
  std::vector<double> C((N + pad) * (M + pad));
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (double &a : A) a = dist(gen);
  for (double &b : B) b = dist(gen);
  for (double &c : C) c = dist(gen);

  std::vector<double> R(C);
  // assume a=rowmajor, b=colmajor
  #pragma omp parallel for
  for (std::size_t m = 0; m < M; ++m) {
    double * const c_ = &R[m * (N + pad)];
    const double * const a_ = &A[m * (K + pad)];
    for (std::size_t n = 0; n < N; ++n) {
      const double * const b_ = &B[n * (K + pad)];
      double sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += a_[k] * b_[k];
      }
      c_[n] += sum;
    }
  }
  std::vector<double> AC(A.size());
  std::vector<double> BR(B.size());
  const std::size_t lk = K+pad;
  for (std::size_t k = 0; k < lk; ++k) {
    const double *ar_ = &A[k];
    const double *bc_ = &B[k];
    double *ac_ = &AC[k * (pad+M)];
    double *br_ = &BR[k * (pad+N)];
    for (std::size_t m = 0; m < (M+pad); ++m) {
      *ac_++ = *ar_;
      ar_ += lk;
    }
    for (std::size_t n = 0; n < (N+pad); ++n) {
      *br_++ = *bc_;
      bc_ += lk;
    }
  }
  std::cout << "done host data preparation" << std::endl;

  GPUMem devA(A.size() * sizeof(double));
  GPUMem devB(B.size() * sizeof(double));
  GPUMem devC(C.size() * sizeof(double));

  Stream stream;
  std::vector<double> T(C.size());
  auto test = [&](bool a_rowmajor, bool b_rowmajor)->void {
    const std::string header = std::string("[A") +
      (a_rowmajor ? 'R' : 'C') + 'B' + (b_rowmajor ? 'R' : 'C') + ']';
    devA.copyToGPU((a_rowmajor ? A : AC).data());
    devB.copyToGPU((b_rowmajor ? BR : B).data());
    devC.copyToGPU(C.data());
    auto start = std::chrono::steady_clock::now();
    dgemm_async(stream.get(), static_cast<const double*>(devA.get()),
      static_cast<const double*>(devB.get()),
      static_cast<double*>(devC.get()), M, N, K,
      a_rowmajor, b_rowmajor,
      (a_rowmajor ? K : M) + pad,
      (b_rowmajor ? N : K) + pad,
      N + pad);
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) {
      throw std::runtime_error("error in kernel execution");
    }
    double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
    std::cout << header << " performance = "
      << (2.0 * M * N * K / sec) << " FLOPS" << std::endl;
    devC.copyToHost(T.data());
    if (T.size() > 3) {
      std::cout << header << "ref.: " << R[0] << ' ' << R[1] << ' ' << R[2] << ' '  << R[3]
        << '\n' << header << "tst.: " << T[0] << ' ' << T[1] << ' ' << T[2] << ' '  << T[3]
        << '\n' << header << "inp.: " << C[0] << ' ' << C[1] << ' ' << C[2] << ' '  << C[3]
        << std::endl;
    }
    double maxdiff = 0;
    unsigned maxprint = 10;
    for (std::size_t i = 0; i < T.size(); ++i) {
      double diff = R[i] - T[i];
      if (diff < 0) diff = -diff;
      if (diff > maxdiff) maxdiff = diff;
      if (diff > 0.02 && maxprint) {
        std::cout << header << " diff at index " << i
          << ", expected = " << R[i] << ", get " << T[i] << std::endl;
        maxprint--;
      }
    }
    std::cout << header << " max difference: " << maxdiff << std::endl;
  };
  test(true, true);
  test(false, false);
  test(true, false);
  test(false, true);
}
