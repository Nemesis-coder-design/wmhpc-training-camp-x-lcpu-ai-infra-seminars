# Assignment 01 答题卡

> GPU 型号：NVIDIA L20  |  Compute Capability：8.9  |  SM 数量：92

## 模块 0：环境准备

### prob 0.1

（运行 `make run/m0_env/01_hello`，多次运行观察输出顺序，记录你的发现）

### prob 0.2

| 项目 | 你的卡 |
|------|--------|
| 型号 / compute capability | NVIDIA L20 / 8.9 |
| SM 数量 | 92 |
| warp 大小 | 32 |
| shared memory / block | 48 KB |
| 最大常驻线程 / SM | 1536 |
| 显存总量 | ~44.4 GiB |

---

## 模块 1：为什么要用 GPU

### prob 1.1

(a) [ 错 ] 对 / 错。理由：GPU的100Tflops是吞吐量，而5GHz是延迟，GPU通常1-2GHz
(b) [ 对  ] 对 / 错。理由：
(c) [ 对 ] 对 / 错。理由：按照阿姆达尔定律，总时间永远取决于串行部分，GPU只能优化并行吞吐
(d) [ 错 ] 对 / 错。理由：延迟不等同于吞吐

<!-- 提示：围绕"延迟 vs 吞吐"这个核心区别来组织理由 -->

### prob 1.2

（从"延迟"和"吞吐"的角度解释：为什么严格在线的串行算法即使总计算量很大，也无法靠 GPU 在几秒内跑完？）

<!-- 提示：串行依赖链的每一步延迟是固定的，吞吐再高也绕不过去 -->
GPU吞吐高，计算快，但串行算法需要等待上一步结果，总耗时应该取决于单步延迟

### prob 1.3

| 执行层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信手段 |
|----------|----------|----------|----------------|----------------|
| thread | kernel 的最小执行单位 | 计算单元上的一个 lane | 自己的寄存器 | （自身天然有序） |
| warp | 32个线程组成的基本调度单位 | SM里面的warp scheduler | 寄存器（线程私有）+warp shuffle（线程束内洗牌） | shuffle指令来让线程交换数据，隐式同步 |
| block / CTA | 一组共同工作线程 | SM本身，一个block固定落在一个SM上 | shared memory（供block内所有线程使用） | __syncthreads |
| grid | 所有block的集合 | GPU整体 | global memory  | 原子操作或kernel启动边界 |

<!-- 提示：阅读 Guide 1.2.1-1.2.2，重点关注 warp 是"一组线程以锁步方式执行"，block 内通过 shared memory + __syncthreads 通信 -->

### prob 1.4

SIMD 与 SIMT 的区别：
SIMD：单指令多数据，必须一起算完
SIMT：单指令多线程，软调度，每个线程是否执行当前指令由硬件调度

Volta 后 branch divergence 不再有性能代价？[ 错 ] 对 / 错。理由：
divergence之后，if-else两条路径仍然需要依次执行完，时间代价损失仍然存在。
<!-- 提示：独立 PC 让每个线程可以独立执行，但 warp 内线程仍然共享 instruction fetch/issue，divergence 时仍需串行执行不同分支 -->

### prob 1.5

（运行 `make run/m1_why_gpu/01_scaling`，填入数据）

| 配置 | 耗时 (ms) | ns / 元素 |
|------|-----------|-----------|
| CPU 单线程 | 13.490 | 3.22 |
| GPU `<<<1,1>>>` | 131.954 | 31.46 |
| GPU `<<<1,256>>>` | 2.125 | 0.51 |
| GPU 铺满 grid | 0.020 | ~0.005 |

(a) GPU 单线程为什么比 CPU 慢这么多？

GPU 核心频率（L20 约 1-2 GHz）远低于 CPU（3-5 GHz），单个 CUDA core 的设计也比 CPU 核心简单得多——没有乱序执行、分支预测等复杂硬件。GPU 把晶体管预算投给了"更多的计算单元"而非"让单个线程跑得更快"，因此单线程延迟约为 CPU 的 10 倍，这是 GPU 架构取舍的直接体现。

(b) 从单 block 到铺满 grid 的提速，说明 GPU 加速计算靠的是什么？

GPU 加速靠的是**大规模并行带来的总吞吐量**。单 block 版（256 线程）比单线程快了约 62 倍；铺满 grid 版（16384 × 256 ≈ 419 万线程）又比单 block 快了约 106 倍。大量线程同时工作，把每个线程的延迟"藏"在了并发的计算和访存请求里——单个线程在等数据时，SM 可以立刻切换到另一个 warp 继续执行。这体现了 GPU 的设计哲学：牺牲单线程延迟，换取总吞吐量。

---

## 模块 2：第一个 CUDA 程序

### prob 2.2

(a) `__global__`  (b) `__device__`  (c)`__host__ __device__`  (d) `__constant__`  (e) `__shared__`

<!-- 提示：__global__, __device__, __host__ __device__, __constant__, __shared__ -->

### prob 2.3

（先跑原版记录耗时，再改完对比）68.2ms
每触发一次 page fault，操作系统/驱动的处理路径大致是：
Copy code to clipboard
GPU 访问不存在的页
  → MMU 产生 page fault
  → 中断 → GPU 驱动捕获
  → 查页表确定数据在哪
  → 发起 PCIe 传输搬这一页
  → 更新页表
  → GPU 恢复执行
  实际优化：驱动不傻，不会只搬一页。GPU 一侧访问某页时，
   驱动会预取相邻的页一起搬（类比 CPU 的 prefetch）。

(a) 为什么 kernel 启动后、CPU 读结果前必须同步？在原先版本里这次同步发生在哪个调用？
保证kernel跑完
原版里同步藏在 cudaMemcpy DeviceToHost里）

(b) 原版耗时：___68.4___ ms  →  改后耗时：___55.9___ ms。分析差距原因：
显式cudamemcpy是阻塞式通信，host线程被阻塞
UM是按需缺页迁移，且与计算重叠，靠warp切换来隐藏page fault延迟，流水线重叠

### prob 2.4

(a) [ 错 ] 对 / 错。理由：异步启动，将kernel加到GPU命令队列后，CPU端立即返回
(b) [ 对 ] 对 / 错。理由：同一stream里面操作严格串行
(c) [ 错 ] 对 / 错。理由：kernel启动时异步的，会在下一个同步点或cuda api调用处报错

<!-- 提示：围绕 kernel launch 异步语义、stream 内保序、sticky error 来组织 -->

### prob 2.5

Bug 原因：超过了max threads / SM 
<!-- 提示：先补上 CUDA_CHECK_KERNEL() 看报错，注意你 prob 0.2 打印过的某个上限 -->

### prob 2.7

grid-stride loop 的价值在哪里？只有 16384 个线程时性能上要付出什么代价？
价值：保证正确性和灵活性，让有限量的线程处理任意大小的数据：一个线程处理多个元素：步长 = gridDim.x * blockDim.x（即 grid 里的总线程数）
代价：每个线程要串行执行 ~1024 次加法，延迟被放大约 1024 倍。

### prob 2.8

(a) block 的执行顺序由谁决定？
GPU 硬件调度器（GigaThread Engine / work distributor）决定，不是程序员。
GPU 拿到一个 grid 后，硬件按自己的策略把 block 分发到各个有空闲资源的 SM 上。哪个 SM 先空出来、哪个 block 先被分配、哪个 SM 上的 block 先跑完——这些都取决于硬件状态（SM 负载、内存访问延迟等），对程序员来说不可预测、不可控制。

(b) 程序的正确性可以依赖 block 的执行顺序吗？这条限制和 scalable programming model 有什么关系？

不能依赖。 这恰恰是 CUDA 的 scalable programming model（可扩展编程模型） 的核心设计：

CUDA 故意让 block 之间完全独立，这样：

透明扩展：你的代码跑在 1 个 SM 的低端卡上和跑在 100 个 SM 的高端卡上，不需要改任何一行。硬件自动把 block 分发给所有可用的 SM，block 越多、SM 越多，并行度越高。如果 block 之间有顺序依赖，这个透明扩展就崩溃了——你必须在代码里知道"当前有多少 SM"和"block 被分到了哪个 SM"，那代码就和硬件绑死了。

无通信 = 无死锁：block 之间没有同步原语（CUDA 不提供 __syncthreads 的跨 block 版本）。block A 不能等待 block B 的结果，因为：

block A 和 B 可能根本不在同一个 SM 上
硬件可能先调度 B 再调度 A，也可能反过来
如果 A 等 B 而 B 还没被调度 → 死锁

---

## 模块 3：SIMT 执行

### prob 3.1
\(\text{linearTid}
=
\text{threadIdx.x}
+
\text{blockDim.x}\times \text{threadIdx.y}
+
\text{blockDim.x}\times \text{blockDim.y}\times \text{threadIdx.z}\)

(a) 线性编号：__43____，第 __ 2____ 个 warp，lane __11____
(b) ___2___ 个 warp
(c) ___2___ 个 warp，浪费在：__尾部 warp 不满____

### prob 3.2

预测：___warp___ 更快，约 __2____ 倍

实测比值：__2.0____

解释：

若两个分支计算量一大一小，两种 kernel 的运行时间分别由什么决定？(TA+TB)和max(TA,TB);

### prob 3.3

(a) 为什么注释掉 sync 后不能正确运行？因为shared memory有些位置的线程可能还未被写入

(b) (选做) 为什么有些位置一直是对的？

### prob 3.4

全 grid 同步的标准做法:

多 Kernel Launch（传统做法）
把需要同步的地方拆成多个 kernel，kernel 返回即隐式同步：
Copy code to clipboard
kernel_phase1<<<grid, block, 0, stream>>>(...);
kernel_phase2<<<grid, block, 0, stream>>>(...);
kernel_phase3<<<grid, block, 0, stream>>>(...);
• ✅ 无 deadlock 风险，兼容所有 GPU
• ✅ stream 内自动保证顺序
• ❌ launch overhead（通常几十 µs）
• ❌ 需要把中间结果写回 global memory

### prob 3.5

interleaved 耗时：__0.0145__ ms，contiguous 耗时：__0.0103__ ms，比值：__1.40x__

性能差距原因：
interleaved addressing 下，warp 内各线程访问 shared memory 的地址跨步为 warp_size × stride，导致 bank conflict——同一 bank 被多个线程同时访问，请求串行化。contiguous addressing 下，相邻线程访问相邻地址（stride=1），无 bank conflict，一个 cycle 完成。这里两版耗时相近（1.40x），WARN 提示可能两版写成了同一个实现——说明 contiguous 版的优化空间已经不大，瓶颈在计算而非访存。

---

## 模块 4：存储空间

### prob 4.1

| 空间 | 谁可见 | 生命周期 | 片上/片外 | 谁管理 |
|------|--------|----------|-----------|--------|
| register | 单个线程 | 线程 | 片上 | 编译器 |
| local | 单个线程 | 线程 | 片外（DRAM） | 编译器 |
| shared | block | 线程块 | 片上 | 手动 |
| global | 所有线程+主机 | 显式分配+释放 | 片外 | 手动 |
| constant | 所有线程+主机 | 全局常量 | 片外 | 手动 |
| L1 / L2 cache | 所有线程 | 硬件自动管理 | 片上 | 硬件 |

<!-- 提示：阅读 Guide 2.3.3-2.3.5、2.3.7 -->

### prob 4.3

constant memory 版本相比 global 版本耗时差距：__0____

constant cache 真正的优势在哪种访问模式？为什么这里测不出差距？
warp 内 32 线程同时读同一地址 1 次 read + 32 路广播
这里瓶颈是读x[i]而不是读这个

### prob 4.4

(a) [ 对  ] 对 / 错。理由：local memory虽然是线程私有，但在片外，通过L1/2 cache加速
(b) [ 对  ] 对 / 错。理由：

### prob 4.6

naive 耗时：__4.0861____ ms，priv 耗时：___0.0248___ ms，提速：___164.8x___

提速来自哪里：
global atomic：~200-400 cycles（L2 cache round-trip + coherence protocol）
shared atomic：~30  cycles（片上 SRAM，单 cycle 访问，无跨 SM 流量）
第一刀：砍掉跨 SM 竞争：  
- 没有 L2 cache 的往返
- 没有跨 SM 的一致性协议

第二刀：降低每次 atomic 本身的延迟
- global atomic 的物理路径:
  SM 核发出请求 → 经过 L1 → 到 L2 cache → 锁 cache line → 
  跨 SM cache coherence 协议沟通 → 读 → 改 → 写 → 解锁 → 
  返回结果到 SM 核
  典型延迟：200-400 GPU cycles

- shared atomic 的物理路径:
  SM 核发出请求 → 片上 SRAM → 
  可能同一 bank 上排队（但 256 bins / 32 banks，冲突少）→ 
  读 → 改 → 写 → 返回结果
  典型延迟：约 30 GPU cycles

### prob 4.7

（运行 `make run/m4_memory/05_bandwidth`，填入数据）

| stride | 1 | 2 | 4 | 8 | 16 | 32 |
|--------|------|------|------|------|------|------|
| GB/s | 707.1 | 457.0 | 616.1 | 979.4 | 997.9 | 735.1 |

趋势分析：

数据呈现一个"先降、再升、再降"的 V 字形，根源是 GPU 的**内存事务（memory transaction）**和 **L2 cache line** 之间的交互。L20 的 L2 cache line 是 128 字节（= 32 个 float），warp 的 32 个线程一次要读 128 字节。带宽取决于"多少次内存事务搬运了多少有效数据"。

**stride=1（连续访问，707.1 GB/s）**：
warp 内 32 个线程访问连续的 32 个 float（128 字节），恰好落在一个 L2 cache line 里。硬件合并为 1 次 128 字节事务 → 100% 有效数据。但为什么没到理论峰值（~864 GB/s）？因为读 + 写双向各占一半带宽，这个 kernel 跑的是读 `in[j]` 和写 `out[i]`，单向峰值约被占满。实测 707.1 GB/s 是读写合并后的有效值。

**stride=2（457.0 GB/s，最慢）**：
warp 内 32 个线程访问 stride=2 的地址：`in[0]`, `in[2]`, `in[4]`...`in[62]`。这 32 个元素总共跨了 64×4=256 字节，需要 2 次 128 字节事务 + 1 次 128 字节事务（跨 cache line 边界），每次事务只有 32×4=128 字节有用。有效带宽减半。此外 stride=2 可能落在同一 bank group 冲突上。

**stride=4（616.1 GB/s）**：
warp 跨度进一步增大到 128×4=512 字节，需要 4 次 128 字节事务。有效数据 32×4=128 字节，带宽利用率 25%。但因为事务次数增多，L2 可以流水处理，实际带宽反而比 stride=2 好。

**stride=8 / 16（979.4 / 997.9 GB/s，最快）**：
这是最反直觉的部分——为什么跳着读反而更快？stride=16 时，32 个线程访问地址跨了 32×16×4=2048 字节，需要 16 次单独的事务（因为地址不连续，每次 warp 请求独立处理），没有 coalescing 优势。**但瓶颈从 L2 带宽变成了 bank-level 并发**：分散的请求可以同时打到不同的 memory channel 上，L2 的多个 bank 并行服务，总吞吐反而接近峰值。这说明 L20 的 memory subsystem 的非连续访问处理能力很强。

**stride=32（735.1 GB/s）**：
stride=32 就是 32×32×4=4096 字节的跨度。每次 128 字节事务只命中一个 float（有效数据 4/128=3.1%），大部分数据被丢弃。L2 事务开销（每条事务的固定延迟）开始主导总时间，带宽回落。再往上到 stride=64/128，会继续恶化，趋近于每次事务只服务一个 float 的极限情况。

**核心教训**：
- coalescing（合并访存）让 stride=1 高效，但不是唯一因素
- 在 L20（Ada Lovelace）这种现代 GPU 上，memory controller 的并发度足够高，适度的分散访问（stride=8/16）反而能打满带宽
- 过度分散（stride=32+）时，每次事务的有效数据太少，带宽浪费在搬运不要的字节上
- "连续访问最快"是旧架构（Fermi/Kepler）的铁律，Ada 时代要实测才知道

### prob 4.8

（运行 `make run/m4_memory/06_occupancy`，填入数据）

| shared memory / block (KB) | 0.0 | 13.2 | 15.0 | 18.0 | 29.0 | 55.0 |
|----------------------------|------|------|------|------|------|------|
| 理论驻留 block / SM | 6 | 6 | 6 | 5 | 3 | 1 |
| occupancy | 100.0% | 100.0% | 100.0% | 83.3% | 50.0% | 16.7% |
| 实测带宽 (GB/s) | 701.9 | 705.3 | 705.3 | 704.9 | 706.8 | 502.0 |

(a) 手算过程（以 shared/block = 29.0 KB 为例）：

已知：
- shared memory / SM = 100 KB（驱动报告值）
- max threads / SM = 1536
- blockDim = 256 threads

由 shared memory 限制的最大驻留 block 数：
  floor(100 KB / 29 KB) = floor(3.448) = 3

由 threads 限制的最大驻留 block 数：
  floor(1536 / 256) = 6

二者取小 → 理论驻留 = 3 blocks/SM

occupancy = 3 × 256 / 1536 = 768 / 1536 = 50.0%  ✓ 与 API 报告一致

(b) 带宽为什么随 occupancy 下降？用"延迟隐藏"解释：

GPU 靠 warp 切换来隐藏显存访问的延迟。当一个 warp 发出内存请求后需要等待 ~200-400 cycles 数据才能到达，SM 会立刻切换去执行另一个就绪的 warp——这就是"延迟隐藏在并发中"。

Little 定律：要达到带宽 B，需要同时在途的字节数 = B × 延迟。
700 GB/s × 300 cycles（L2 延迟，~1.5 GHz）≈ 140,000 字节同时在途。
每个 warp 提供 1 个 128B cache line 请求。
140,000 / 128 ≈ 1094 个并发请求 → 需要 ~34 个 warp 同时在飞。

occupancy 就是 SM 上实际驻留的 warp 比例。16.7% occupancy = 只有 48×0.167 = 8 个 warp 驻留。8 个 warp 远远不够维持 1000+ 个在途请求，显存流水线出现空闲——带宽下降。

(c) 从 100%→50% 带宽掉了多少？从 50%→16.7% 又掉了多少？解释差别：

100% → 50%：带宽从 705.3 到 706.8 GB/s，**几乎为零**。这是本次实验最反直觉的发现。

理由：L20 使用的 GDDR6 显存单次访问延迟（~150-200 cycles）远低于 A100 的 HBM2e（~250-400 cycles）。低延迟意味着只需要更少的并发 warp 就能填满 memory pipeline。在 50% occupancy（24 个 warp / SM）时，活跃 warp 数量仍然远超维持 GDDR6 峰值带宽所需的下限——带宽被 memory controller 数量、GDDR6 burst length 等硬件瓶颈限死在 ~705 GB/s，而非 warp 数量。

50% → 16.7%：带宽从 706.8 跌到 502.0 GB/s，掉了 **29%**。此时每个 SM 只有 8 个 warp（16.7% × 48），在途请求数终于跌破了维持峰值带宽所需的阈值。8 个 warp 在 300 cycle 往返延迟中只能维持 ~8 个并发请求，远不足以填满 92 个 SM 的 memory controller 流水线。

对比 A100 的教科书表现（在 50% occupancy 时带宽就已经明显下降），L20 的"平顶"更宽——因为它用的是低延迟的 GDDR6，而不是高延迟高带宽的 HBM。这是架构选择（推理优化 vs 训练优化）在性能特征上的直接体现。

---

## 模块 5：计时与异步

### prob 5.1

（运行 `make run/m5_async/01_timing_trap`，记录三个数值）

| 方式 | 数值 |
|------|------|
| host 不等 GPU | 0.0035 ms |
| host 等 GPU | 0.1193 ms |
| cudaEvent | 0.1175 ms |

(a) 哪个能当 kernel 耗时？ **cudaEvent（0.1175 ms）**

(b) 另外两个各测的是什么？

- host 不等 GPU（0.0035 ms）：测的是 **kernel launch 本身的开销**——host 把 kernel 提交到 GPU 命令队列、立即返回、不等它执行完。这就是 `<<<>>>` 异步启动的代价，约 3.5 微秒。

- host 等 GPU（0.1193 ms）：测的是 **host 视角的完整时间**——从 kernel launch 到 `cudaDeviceSynchronize` 返回。包含三部分：launch 开销 + kernel 实际执行 + host 端 sync 的系统调用开销。0.1193 - 0.1175 = 0.0018 ms（1.8 微秒），这 1.8 微秒就是 launch + sync 的额外开销。

### 核心概念

```
时间线：

host:  [launch kernel 3.5µs]  [立刻返回继续执行]  ........  [cudaDeviceSync等待]  [返回]
GPU:                          [kernel 执行 117.5µs]                    ↑
                                                                  同步点

host不等GPU: 只测了 [launch kernel] 这一段 = 0.0035 ms
host等GPU:   测了 [launch] + ...等待... + 直到 GPU 完成 = 0.1193 ms  
cudaEvent:   GPU 硬件在命令流里打的两个时间戳之差 = 0.1175 ms  ← 纯 kernel 执行时间
```

cudaEvent 最准，是因为它把时间戳记录在 **GPU 的命令流里**——start 和 stop 事件都是 GPU 硬件在 kernel 前后打的时间戳，不受 host 端调度、系统调用、CPU 中断等干扰。host 的 `std::chrono` 测量的是 CPU 这端的墙钟时间，中间夹了各种 OS 层面的不确定性。

### 实践教训

永远用 `cudaEvent` 测 GPU 性能。`std::chrono` 不用 sync 会以为 kernel 在 0.0035 ms 内完成了（其实没有），用 sync 则混入了系统调用的噪声。这就是为什么 2.9（SAXPY）要求你手写 cudaEvent 计时——它的 0.1 ms 级 kernel 用两种方式可能差出几十倍。

### prob 5.2

(a) [ 对 ] 对 / 错。理由：stream就像一条传送带，按顺序执行同一stream的指令
(b) [ 对 ] 对 / 错。理由：异步默认语义:CPU把活交给GPU，立即进入下一行，除非主动用cudaDeviceSynchronize
(c) [ 对 ] 对 / 错。理由：uM下，相当于虚拟大仓库，GPU在用某页，CPU访问，就会发生page fault

---

## 模块 6：Tile 视角

### prob 6.1

(a) [ 错 ] 对 / 错。理由：它是数据搬运和计算的单位
tile 不是一块物理内存区域，而是一个逻辑概念——它是"一个 block 要处理的那一小块数据"的抽象。它描述的是：把一个大数组切成小块，每一块（tile）由 shared memory 暂存，被 block 内的线程协作处理。


矩阵 1024×1024 → 切成 64×64 的 tile（逻辑划分）

┌────┬────┬────┬────┐
│Tile│Tile│Tile│ ... │
│ 0  │ 1  │ 2  │     │
├────┼────┼────┼────┤
│Tile│    │    │     │
│ 16 │    │    │     │
├────┼────┼────┼────┤
│... │    │    │     │
└────┴────┴────┴────┘

每个 tile 的实际数据可能暂存在：
  - global memory（源数据）
  - shared memory（搬运进来后，低延迟访问）
  - 寄存器（正在被计算的个别元素）
(b) [ 对 ] 对 / 错。理由：描述的是一个block对数据做什么，thread的分工由编译器决定
(c) [ 错 ] 对 / 错。理由：tile模型是建立在SIMT之上

### prob 6.2

| | CUDA SIMT | cuTile | Triton |
|---|-----------|--------|--------|
| 并行单位 | block 里的 thread | **block**（tile） | **program**（类似 block） |
| 编号 | blockIdx / threadIdx | **ct::bid(0)**（只有 block 编号，无 thread 编号） | **tl.program_id()**（只有 program 编号，无 thread 编号） |
| 数据分工 | 线程用全局下标来划分数据（手写 `threadIdx.x + blockIdx.x * blockDim.x`） | **tile view**——定义 tile 形状，编译器自动把 tile 元素分配到线程 | **offsets + mask**——用 `tl.arange` 生成偏移，声明式处理一段数据 |
| 边界处理 | if 判断（手写 `if (idx < n)`） | **隐式**——tile view 的 load/store 自动处理越界 | **mask**——`offsets < n`，load/store 传 mask 参数跳过越界位置 |

### prob 6.3

(a) "每个线程对应哪个/哪些元素"由谁决定？

由**编译器**决定。cuTile 模型只让你定义"这个 block 要处理哪个 tile 的数据"（通过 `a_view.load((bid,))`），tile 内部的元素如何分配到具体线程——哪些元素被 thread 0 处理、哪些被 thread 1 处理——完全由编译器在编译时根据 tile 形状和硬件参数自动推导。程序员不再需要写 `threadIdx.x + blockIdx.x * blockDim.x`。这对标的是 CUDA SIMT 中"每个线程通过全局下标手动认领自己要处理的元素"的模式——cuTile 把这块工作从"用户写的代码"变成了"编译器生成的代码"。

(b) CUDA SIMT 版里出现、这里完全没体现的概念：

1. **threadIdx / blockDim**：cuTile 不暴露线程编号和 block 尺寸。没有 `threadIdx.x`，也没有 `blockDim.x`。程序员不知道一个 block 里有几个线程——这是编译器的内部决策。

2. **手写的全局下标**：`int idx = threadIdx.x + blockIdx.x * blockDim.x`。在 cuTile 里，这个公式消失了——`a_view.load((bid,))` 和 tile 形状一起隐式完成了定位。

3. **手写的边界保护**：`if (idx < n)`。cuTile 的 `tiled_view` 和 `load`/`store` 自动处理越界——最后一小块数据不足 TILE 个元素时，编译器自动生成边界检查代码或填 0。

4. **显式的 launch 配置**：`<<<blocksPerGrid, threadsPerBlock>>>`。cuTile 里你只需要定义 tile 的形状 `(TILE,)`，grid 的维度由数组大小和 tile 形状自动推导。block 内线程数也是编译器选的。

5. **显式的同步**：`__syncthreads()`。cuTile 内部如果需要同步（比如 tile load 完成后才能计算），编译器会自动插入——程序员不需要写。

这五个概念的消失不是巧合——它们在 CUDA SIMT 中都是**"如何把工作分配给线程"**的细节。cuTile 的设计哲学是：这些细节是机械的、可推导的，应该由编译器处理；真正需要程序员决策的是 tile 大小（影响 occupancy 和带宽利用率），所以只留了 `TILE` 这个参数。

---

## 模块 7：TileLang 与 Triton

### prob 7.2

与 Module 2 改 CUDA kernel 相比，这次的改动集中在什么部分？主体代码为什么不用动？

### prob 7.4

2.6 的四个空（行号、列号、边界保护、grid 尺寸）哪些在这里还有对应？没有对应的那个去哪了？

### prob 7.5

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
|--------|-----------|--------|--------|----------|
| 线程到数据的映射 | 用户 | | | |
| 边界处理 | 用户 | | | |
| tile / block 尺寸的选择 | 用户 | | | |
| block 内同步 | 用户 | | | |

### prob 7.6

五个空涉及 shared memory、寄存器 tile、流水，而 Triton 版 matmul 并未显式指定——为什么？

---

## 模块 8：平台与编译

### prob 8.1

(a) [ 错 ] 对 / 错。理由：PTX 是中间表示（Intermediate Representation），不是机器码。GPU 直接执行的是 SASS——由 PTX 经 ptxas 编译或驱动 JIT 生成的机器码。
(b) [ 错 ] 对 / 错。理由：SASS 是架构相关的机器码，sm_70 的指令只能在 Volta 架构的卡上执行。compute capability 9.0 的卡需要 sm_90 的 SASS（或至少携带 PTX 由驱动 JIT）。
(c) [ 对 ] 对 / 错。理由：fatbin（fat binary）的设计目标就是携带多份代码——可以同时嵌入 sm_70、sm_80、sm_90 等多个架构的 SASS，外加一份 PTX 作为 fallback。驱动加载时自动选择最佳匹配。
(d) [ 对 ] 对 / 错。理由：当可执行文件中没有匹配当前 GPU 的 SASS、但携带了 PTX 时，GPU 驱动会在运行时调用 ptxas 将 PTX JIT 编译为当前卡的机器码并缓存。这是 CUDA 向前兼容的关键机制。

### prob 8.2（选做）

(a) SASS-only 运行结果 / 报错信息：

(b) PTX-only 能否运行？PTX 在什么时候、由谁编译成机器码？

### prob 8.3（选做）

Runtime API 与 Driver API 各自定位。`cudaMalloc` 属于哪个？

---

## Bonus：matmul（选做）

### (a) Naive CUDA

| BS | 耗时 (ms) | GFLOPS |
|----|-----------|--------|
| 8 | | |
| 16 | | |
| 32 | | |

### (b) Tiled Triton

（运行 `python -c "from kernels.matmul_triton import bench; bench()"`）

| BLOCK_M | BLOCK_N | BLOCK_K | 耗时 (ms) | TFLOPS |
|---------|---------|---------|-----------|--------|
| | | | | |

cuBLAS: ______ ms, ______ TFLOPS

### (c) TileLang

（运行 `python -c "from kernels.tilelang_matmul import bench; bench()"`）

| block_M | block_N | block_K | threads | stages | 耗时 (ms) | TFLOPS |
|---------|---------|---------|---------|--------|-----------|--------|
| | | | | | | |

三种实现性能差距原因分析（特别是 TileLang 比 Triton 控制粒度更细，为什么反而略慢）：
