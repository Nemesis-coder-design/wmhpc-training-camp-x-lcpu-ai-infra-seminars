"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T


_softmax_kernel_cache = {}


@tilelang.jit
def make_softmax(M: int, N: int):
    # N > 0 时，得到不小于 N 的最小 2 的幂。
    # 例如：300 -> 512，1024 -> 1024，4096 -> 4096。
    PADDED_N=1 << (N - 1).bit_length()

    @T.prim_func
    def softmax_kernel(
        A: T.Buffer((M, N), "float32"),
        B: T.Buffer((M, N), "float32"),
    ):
        # 启动 M 个 block，每个 block 处理 A 的一行。
        with T.Kernel(M, threads=128) as row_idx:
            # 实际 fragment 宽度使用 PADDED_N，而不是 N。
            row = T.alloc_fragment((PADDED_N,), "float32")
            max_val = T.alloc_fragment((1,), "float32")
            sum_exp = T.alloc_fragment((1,), "float32")

            # 加载当前行。
            # 对超出 N 的填充位置写入 -inf，使其不影响最大值。
            for col_idx in T.Parallel(PADDED_N):
                row[col_idx] = T.if_then_else(
                    col_idx < N,
                    A[row_idx, col_idx],
                    -T.infinity("float32"),
                )

            # max_val[0] = max(row)
            T.reduce_max(row, max_val, dim=0)

            # 数值稳定的指数计算：
            # exp(x_i - max(x))
            #
            # 填充位置原来是 -inf，因此 exp(-inf - max_val) = 0，
            # 后续不会影响求和。
            for col_idx in T.Parallel(PADDED_N):
                row[col_idx] = T.exp(row[col_idx] - max_val[0])

            # sum_exp[0] = sum(exp(x_i - max(x)))
            T.reduce_sum(row, sum_exp, dim=0)

            # 只写回真实的 N 个元素，不写 padded 部分。
            for col_idx in T.Parallel(N):
                B[row_idx, col_idx] = row[col_idx] / sum_exp[0]

    return softmax_kernel


def softmax(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("softmax expects a torch.Tensor")

    if x.ndim != 2:
        raise ValueError(
            f"softmax expects a 2D tensor, but got shape {tuple(x.shape)}"
        )

    if not x.is_cuda:
        raise ValueError("softmax expects a CUDA tensor")

    if x.dtype != torch.float32:
        raise TypeError(
            f"softmax expects torch.float32, but got {x.dtype}"
        )

    M, N = x.shape

    if N > 4096:
        raise ValueError(f"softmax only supports N <= 4096, but got N={N}")

    # 空矩阵无法执行行内归约，直接返回同形状空结果。
    if M == 0 or N == 0:
        return torch.empty_like(x)

    # T.Buffer 按连续二维数组访问，因此对非连续输入先连续化。
    x = x.contiguous()
    y = torch.empty_like(x)

    # 形状决定了生成的 Buffer 形状和 PADDED_N，因此按形状缓存。
    # device 也放入 key，避免多 GPU 环境中复用错误的编译结果。
    device_index = x.device.index
    cache_key = (device_index, M, N)

    kernel = _softmax_kernel_cache.get(cache_key)
    if kernel is None:
        kernel = make_softmax(M, N)
        _softmax_kernel_cache[cache_key] = kernel

    kernel(x, y)
    return y