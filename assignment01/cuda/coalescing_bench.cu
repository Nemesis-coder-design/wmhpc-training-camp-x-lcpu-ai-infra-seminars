// coalescing 微基准测试 — 测试 L20 上不同访存模式的带宽
// 编译：nvcc -O2 -std=c++17 -arch=native -o coalescing_bench coalescing_bench.cu
// 运行：./coalescing_bench
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error at " << __FILE__             \
                  << ":" << __LINE__ << ": "                  \
                  << cudaGetErrorString(err) << "\n";         \
        std::exit(1);                                         \
    }                                                         \
} while (0)

// ====== 实验 1：不同 stride 的读取带宽 ======
// 每个 warp 的 32 个线程读 32B/64B/128B segment，stride 控制分散程度
__global__ void strided_read(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = ((long)i * stride) & (n - 1);  // n 是 2 的幂，& (n-1) 等价取模
        out[i] = in[j];
    }
}

// ====== 实验 2：向量化读取（float / float2 / float4） ======
__global__ void vec1_read(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void vec2_read(const float2 *in, float2 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void vec4_read(const float4 *in, float4 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// ====== 实验 3：warp 内线程到缓存的映射 ======
// 32 个线程同时访问 32 个地址，这些地址可能落在：
//   - 同一个 128B cache line（最优 coalescing）
//   - 不同 cache line（分散，测试 L2 bank 并行度）
//
// 控制参数 shift：地址偏移 = threadIdx.x << shift
//   shift=2 → 连续 4B 间隔 → 128B 连续段
//   shift=5 → 32B 间隔 → 跨 1024B
//   shift=7 → 128B 间隔 → 每个线程独占一个 cache line
__global__ void warp_level_stride(const float *in, float *out, int n, int shift) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) & (n - 1 - 31);
    int offset = (threadIdx.x << shift) & (n - 1);
    int idx = (base + offset) & (n - 1);
    out[idx] = in[idx];
}

// ====== 实验 4：跨 warp 的 bank conflict 敏感度 ======
// 测试 warp 内 32 线程访问同一 128B segment 的不同偏移位置
__global__ void misaligned_read(const float *in, float *out, int n, int misalign_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 为越界偏移预留空间：n 会比 in 的实际大小小一点
    if (i < n - 128) {
        // 故意错位：从 misalign_bytes 字节偏移开始读
        const char *in_bytes = (const char *)in + misalign_bytes;
        float val = *(const float *)(in_bytes + (long)i * sizeof(float));
        out[i] = val;
    }
}

// ====== 辅助函数 ======
static cudaEvent_t t_start, t_stop;

static void init_timer() {
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));
}

static double measure_ms(void (*fn)(), dim3 grid, dim3 block,
                         int reps = 50) {
    // 热身
    for (int r = 0; r < 5; r++) fn<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t_start));
    for (int r = 0; r < reps; r++) fn<<<grid, block>>>();
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
    return ms / reps;
}

// ====== 主程序 ======
int main() {
    // 打印设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name
              << " | CC: " << prop.major << "." << prop.minor
              << " | SMs: " << prop.multiProcessorCount
              << " | L2: " << prop.l2CacheSize / 1024 << " KB"
              << " | BW: " << (prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2 / 1e6)
              << " GB/s\n\n";

    const int n = 1 << 24;  // 16M floats = 64 MB
    const size_t bytes = n * sizeof(float);
    int threads = 256, blocks = (n + threads - 1) / threads;
    init_timer();

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 0, bytes));

    // ==========================================================
    std::cout << "═══════════ 实验 1：stride 对带宽的影响 ═══════════\n";
    std::cout << "原理：strided_read kernel，读写各 64MB\n";
    std::cout << "stride  耗时(ms)  有效BW(GB/s)  cache line效率\n";
    std::cout << "────────────────────────────────────────────────\n";

    int strides[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128};
    for (int s : strides) {
        const int reps = (s <= 4) ? 30 : 20;

        // 手动计时
        for (int r = 0; r < 5; r++)
            strided_read<<<blocks, threads>>>(d_in, d_out, n, s);
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms_f;
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int r = 0; r < reps; r++)
            strided_read<<<blocks, threads>>>(d_in, d_out, n, s);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_f, t_start, t_stop));
        double ms = ms_f / reps;

        // 有效带宽 = 读 64MB + 写 64MB = 128MB
        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;
        // 效率粗略估算：每个 128B cache line 含 32 个 float，
        // 实际命中 = min(32, 128 / (stride * 4))
        int useful_per_line = (int)(128.0 / (s * 4));
        if (useful_per_line > 32) useful_per_line = 32;

        std::cout << std::setw(3) << s
                  << std::setw(11) << std::fixed << std::setprecision(4) << ms
                  << std::setw(12) << std::setprecision(1) << bw
                  << std::setw(15) << useful_per_line << " floats/128B\n";
    }

    // ==========================================================
    std::cout << "\n═══════════ 实验 2：向量化读取对比 ═══════════\n";
    std::cout << "原理：1 条 float4 load = 4 条 float load 的数据量\n";
    std::cout << "方式      耗时(ms)  有效BW(GB/s)  每次load字节\n";
    std::cout << "────────────────────────────────────────────\n";

    int n1 = n, n2 = n / 2, n4 = n / 4;
    int blocks2 = (n2 + threads - 1) / threads;
    int blocks4 = (n4 + threads - 1) / threads;

    auto test_vec = [&](const char *label, double ms, int load_bytes) {
        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;
        std::cout << std::setw(6) << label
                  << std::setw(12) << std::fixed << std::setprecision(4) << ms
                  << std::setw(12) << std::setprecision(1) << bw
                  << std::setw(12) << load_bytes << "\n";
    };

    { // float
        for (int r = 0; r < 5; r++) vec1_read<<<blocks, threads>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int r = 0; r < 50; r++) vec1_read<<<blocks, threads>>>(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
        test_vec("float", ms / 50, 4);
    }
    { // float2
        for (int r = 0; r < 5; r++) vec2_read<<<blocks2, threads>>>((float2 *)d_in, (float2 *)d_out, n2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int r = 0; r < 50; r++) vec2_read<<<blocks2, threads>>>((float2 *)d_in, (float2 *)d_out, n2);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
        test_vec("float2", ms / 50, 8);
    }
    { // float4
        for (int r = 0; r < 5; r++) vec4_read<<<blocks4, threads>>>((float4 *)d_in, (float4 *)d_out, n4);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int r = 0; r < 50; r++) vec4_read<<<blocks4, threads>>>((float4 *)d_in, (float4 *)d_out, n4);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
        test_vec("float4", ms / 50, 16);
    }

    // ==========================================================
    std::cout << "\n═══════ 实验 3：warp 内线程间距对合并的影响 ═══════\n";
    std::cout << "同一个 warp，32 线程的地址间距 = 1 << shift 字节\n";
    std::cout << "shift    间距(B)  跨段数   耗时(ms)  BW(GB/s)\n";
    std::cout << "──────────────────────────────────────────────\n";

    for (int shift = 2; shift <= 8; shift++) {
        for (int r = 0; r < 5; r++) warp_level_stride<<<blocks, threads>>>(d_in, d_out, n, shift);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int r = 0; r < 30; r++) warp_level_stride<<<blocks, threads>>>(d_in, d_out, n, shift);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
        ms /= 30;

        int gap = 1 << shift;                    // 相邻线程地址间距 (bytes)
        int total_span = gap * 32;               // 整个 warp 跨多少字节
        int segments = (total_span + 127) / 128; // 需要多少个 128B segment
        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;

        std::cout << std::setw(3) << shift
                  << std::setw(11) << gap
                  << std::setw(9) << segments
                  << std::setw(11) << std::fixed << std::setprecision(4) << ms
                  << std::setw(10) << std::setprecision(1) << bw << "\n";
    }

    // ==========================================================
    std::cout << "\n═══════ 实验 4：地址对齐对合并的影响 ═══════\n";
    std::cout << "从起始地址往后再偏移 misalign 字节开始读取\n";
    std::cout << "偏移(B)  耗时(ms)  BW(GB/s)\n";
    std::cout << "──────────────────────────────\n";

    for (int off = 0; off <= 124; off += 4) {
        for (int r = 0; r < 5; r++) misaligned_read<<<blocks, threads>>>(d_in, d_out, n, off);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int r = 0; r < 30; r++) misaligned_read<<<blocks, threads>>>(d_in, d_out, n, off);
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
        ms /= 30;
        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;

        std::cout << std::setw(4) << off
                  << std::setw(11) << std::fixed << std::setprecision(4) << ms
                  << std::setw(10) << std::setprecision(1) << bw;

        // 标记是否跨 128B 边界
        if (off == 0)  std::cout << "  ← 128B 对齐";
        if (off == 124) std::cout << "  ← 恰好跨 128B 边界";
        std::cout << "\n";
    }

    // 清理
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
