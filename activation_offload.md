# 激活（Activation）CPU Offload 设计文档（ms-swift）

本文档梳理 `swift/callbacks/activation_cpu_offload.py` 中激活 CPU Offload 的 **动机、架构、核心实现原理、价值与权衡、以及技术难点**。

## 动机（Motivation）

大模型（LLM/MLLM）训练经常受限于 **激活显存（activation memory）**：

- 即使使用了 FSDP/ZeRO 类的参数/梯度/优化器状态分片，**反向需要保存的中间激活（saved for backward）** 仍可能成为显存峰值的主要来源。
- 激活检查点（activation checkpointing / gradient checkpointing）可以显著降低激活显存，但会引入重算开销。
- CPU offload 的目标是：把 “saved-for-backward” 张量从 GPU 迁移到 CPU，降低 GPU 峰值显存；并尽量通过 **计算与拷贝重叠** 来减少性能损失。

该实现主要面向常见场景：

- 使用 FSDP/FSDP2 训练（大量 transformer block 被 wrap），激活显存限制 batch size / seq len。
- 允许额外 CPU 内存与 PCIe/NVLink 传输开销，以换取更大的 GPU 显存余量。

## 目标与非目标（Goals / Non-goals）

**目标**

- forward 时将 saved-for-backward 的张量从 GPU offload 到 CPU。
- backward 需要这些张量前将其 reload 回 GPU。
- 提供可选的 **异步** offload/reload 路径，尽量与计算重叠。
- 通过 Trainer callback + `fsdp_config` 小开关，方便在 FSDP/FSDP2 下启用。

**非目标**

- 通用的任意张量交换（只聚焦 autograd 保存的张量）。
- 对所有模型结构都给出最优调度（这里假设“层状/块状”结构）。
- 在该文件中支持除 FSDP/FSDP2 之外的训练策略（代码中显式断言）。

## 核心实现原理（Core Principle）

PyTorch autograd 提供了对 “saved tensors” 的 hook 能力：

- forward 阶段，很多算子会调用 `ctx.save_for_backward(tensor)` 保存张量；
- backward 阶段，autograd 会取回这些张量以计算梯度。

本实现利用：

`torch._C._autograd._push_saved_tensors_default_hooks(save_hook, load_hook)`

做到：

1. 拦截 `save_for_backward`：把本应保存的 tensor **替换为一个轻量 tag（标识符）**；
2. 把真实 tensor 的数据存到别处（CPU 副本，或先登记后批量 offload）；
3. backward 取回时：根据 tag 恢复真实 tensor（必要时先 reload 回 GPU）。

这个“拦截 saved tensor”的思路，使得在 **不改模型代码** 的情况下实现激活 offload 成为可能。

## 架构总览（Architecture）

实现可以分成三层：

1. **Hook 封装（上下文管理器）**
   - `CpuOffloadHookWithOffloadHandler`：安装/卸载 saved-tensor hooks；具体如何 offload/reload 交给 handler。

2. **Offload Handler（策略 + 存储/调度）**
   - `OffloadHandler` 接口：`tensor_push()` / `tensor_pop()`。
   - 两个实现：
     - `SynchronizedGroupOffloadHandler`：同步（阻塞）拷贝，简单可靠。
     - `AsyncDoubleBufferGroupOffloadHandler`：异步批量 offload/reload，使用 stream + 双缓冲以覆盖拷贝开销。

3. **模型集成**
   - `ActivationHandler`：wrap 每个 transformer/FSDP-wrapped module 的 forward：
     - 进入 offload 上下文（开启 saved-tensor hooks）；
     - 执行原 forward（可选 checkpoint）；
     - 执行一次 “commit” op 推进 offload 调度。
   - `ActivationCpuOffloadCallBack`：训练开始时根据配置启用。

## 关键组件（Key Components）

### 1) `CpuOffloadHookWithOffloadHandler`

职责与功能（更细化）：

- `__enter__`：安装 saved-tensor hooks：
  - `on_save_for_backward(tensor)` → `handler.tensor_push(tensor)` 返回一个标识符；
  - `on_get_saved_tensor(id)` → `handler.tensor_pop(id)` 返回真实 tensor。
- `__exit__`：卸载 hooks。

要点：

- hook 不关心策略细节；同步/异步/分组/预取等都由 handler 实现。
- “保存的对象”在 autograd 里会被替换为 `tensor_push` 返回的 **retrieve_identifier**（可以是 tuple/int/任意可 picklable 的小对象），因此 handler 需要保证：
  - `tensor_push` 返回的标识符在整个 forward→backward 生命周期内可用于索引；
  - `tensor_pop` 能在 backward 中按需返回与原张量 **同 dtype/shape/device 语义一致** 的 tensor（通常为原 device 上的 tensor）。
- 该 hook 的作用范围由上下文控制：`ActivationHandler.pre_forward()` 进入上下文，`post_forward()` 退出上下文，因此 offload 只覆盖被 wrap 的 module forward 执行期间产生的 saved tensors。

### 2) 以 “offload group” 为单位的调度

现实中的关键难点是时序调度：

- offload 太早：拷贝/同步会阻塞计算；
- offload 太晚：显存峰值降不下来。

该实现引入 **group**（近似“按层分组”）：

- 每个被 wrap 的 module forward 结束都会插入一次 **commit 点**；
- handler 依据 commit 点进行：
  - 旧 group 的 bulk offload；
  - 释放旧 group 的 GPU 引用（让显存可回收）；
  - backward 前的 bulk reload / 预取。

更具体地讲，本实现把“一个 module forward 期间产生的 saved tensors”归入当前 `current_group`，并通过 commit 推进 group：

- `current_group`：当前组编号（forward 递增，backward 递减）。
- `tensor_count_current_group`：当前组内 saved tensor 计数，用于生成稳定 tag。
- tag 形式通常为 `(group_id, tensor_idx)`：既能区分组，又能在组内稳定索引。

### 3) `GroupCommitFunction`（dummy autograd op）

`GroupCommitFunction` 输出与输入相同，但会回调 handler：

- forward：`on_group_commit_forward()`
- backward：`on_group_commit_backward()`

这给 handler 提供两个确定的“时间点”：

- 每层 forward 后的确定事件；
- backward 过程中对应位置的确定事件。

为什么必须用“autograd Function”而不是普通 Python 调用：

- handler 需要在 **forward 和 backward 都被调用**，并且调用时机要与 autograd 拓扑一致；
- 用 `torch.autograd.Function` 可以把“提交点”插入计算图，保证 backward 过程能触发对称的 `on_group_commit_backward()`；
- 由于该 op “输入=输出”，不会改变模型数值语义，只提供调度信号。

### 4) `SynchronizedGroupOffloadHandler`（同步版本）

机制：

- `tensor_push`：
  - 生成 `(group_id, tensor_idx)` tag；
  - 立刻分配 CPU buffer 并从 GPU copy 到 CPU（尽可能使用 pinned + non_blocking）；
  - state 存为 `(device, cpu_backup)`。
- `tensor_pop`：
  - 把 CPU backup copy 回原 device（`cpu_backup.to(dev)`）。

优点：

- 简单、稳定。

缺点：

- 拷贝与计算在同一 stream（等价于阻塞），性能损失可能明显。

NPU 说明：

- NPU 对 pinned + 异步拷贝支持不完整，因此走同步 copy。

适用场景：

- 用于验证正确性、或在异步流/同步点很难调的环境下作为保底方案；
- 当模型每层计算很短、通信无法被覆盖时，同步版本可能与异步版本性能接近但更稳定。

### 5) `AsyncDoubleBufferGroupOffloadHandler`（异步双缓冲）

这是主要的性能导向实现。

核心点：

- 使用两个专用 stream：
  - `d2h_stream`：device→host；
  - `h2d_stream`：host→device。
- 维护关键映射：
  - `tensor_tag_to_state`：tag →（GPU tensor）或（轻量 `(key, shape)` 指向 CPU 存储）；
  - `group_offload_mapping`：group_id → `key -> (device, cpu_backup)`（bulk offload 的 CPU 存储）；
  - `tensor_tag_to_buf`：tag → GPU 引用（用于控制何时释放 GPU 侧引用）。
- 在 group 边界做 bulk offload/reload，降低 per-tensor 频繁调度开销。

双缓冲行为：

- 计算 group `i+1` 时，可以在 `d2h_stream` 上 offload group `i`；
- backward 时在真正需要前，提前在 `h2d_stream` 上 reload。

职责与关键数据结构（更细化）：

- `tensor_tag_to_state`：
  - forward 保存时先登记 GPU tensor；
  - 当某个 group 被 bulk offload 后，把对应条目替换为 `(key, shape)`，其中 `key` 用 `_get_unique_tensor_key` 去重（避免重复拷贝同一 storage），`shape` 用于 reload 后 view 回原形状。
- `group_offload_mapping[group_id]`：
  - 保存该组 bulk offload 的去重映射：`key -> (device, cpu_backup)`；
  - reload 时把 `cpu_backup` `.to(device)` 得到 GPU tensor，并可按 `shape` view 回去。
- `tensor_tag_to_buf`：
  - 持有需要延迟释放的 GPU 引用（否则 Python 引用丢失后可能过早释放/影响调度窗口）；
  - 在合适的同步点把旧 group 的引用置为 `None`，让显存更早可回收。
- `offloaded_group_count`：
  - 追踪当前已经推进到哪个 offload group，用于决定下一次 bulk offload / bulk reload 的目标 group。

同步与正确性保障（要点）：

- `d2h_stream.wait_stream(current_stream)` / `current_stream.wait_stream(d2h_stream)`：
  - 保证 offload 读取到的是 forward 计算完成后的 tensor；
  - 并在需要释放/进入下一窗口时确保拷贝完成，避免访问未完成的数据。
- backward 侧使用 `h2d_stream` 预取，并在 commit 点与当前 stream 做双向 wait，确保 `tensor_pop()` 时对应 group 已经 reload 完成。

窗口策略（`layer_window_map`）：

- 构造 “group_idx → 需要同步的 layer id” 映射，尝试把 offload 负载均匀分摊到 forward 过程；
- 目标是：
  - 更稳定地利用 CPU/GPU 互连带宽；
  - 避免某个时间点突然同步导致大 stall；
  - 理想情况下 GPU 同时保留的 group 很少（设计意图是“最多两组”）。

### 6) `FSDPParameterFilter`

为什么需要：

- saved-tensor hook 看到的张量不一定都是激活（可能包括参数相关张量）。

做什么：

- 记录模型参数的 storage 指针集合；
- `__call__` 只有在 tensor 的 storage 不属于参数集合时才返回 True（认为是可 offload 激活）。

价值：

- 避免误 offload 参数，减少无意义 CPU 传输。

细节：

- 该过滤器通过 `tensor.untyped_storage().data_ptr()` 与参数 tensor 的 storage 指针集合比对；
- 每次进入 module forward 前，`ActivationHandler.pre_forward()` 会调用 `update_model_parameters(module)` 更新集合，避免 FSDP 动态扁平化/重建参数导致指针变化时误判。

### 7) 模型集成：`enable_activation_offloading`

该功能仅在 FSDP/FSDP2 下启用：

- 遍历模型，收集 FSDP/FSDP2 wrapped modules（并排除 embedding 这种激活很小的情况）；
- 通过 `get_activation_offload_context()` 创建 offload 上下文与 commit 函数；
- 用 `ActivationHandler.wrap_module_forward_method()` wrap 每个 module 的 forward。

与 checkpoint 的兼容：

- Transformers 自带的 gradient checkpointing 在某些策略下与 saved-tensor hooks 有冲突；
- 若 `enable_ckpt` 开启，则禁用 HF checkpointing（若存在对应 API），并在 `ActivationHandler` 内部用
  `torch.utils.checkpoint.checkpoint(use_reentrant=True)` 实现另一种 checkpoint，以便与 offload 共存。

更细节的执行形态：

- “找层”：递归遍历 `named_children()`，只要遇到 FSDP/FSDP2 wrapper 就把该 wrapper 视为一个 group（并收集到 `layers`）。
- “wrap forward 的对象”：
  - FSDP1：实际 wrap `child._fsdp_wrapped_module`（底层真实 module）；
  - FSDP2：直接 wrap 该 module（DTensor/FSDP2 的 wrapper 结构不同）。
- “分组数量”：
  - `get_activation_offload_context(len(layers) - 1, len(layers), tensor_filter)`：默认 offload 的 group 数小于层数 1，
    目的是保留最后一组（最靠近 loss 的激活）在 GPU 上，减少 backward 首段的 reload 压力。

### 8) Trainer 回调：`ActivationCpuOffloadCallBack`

启用路径：

- `on_train_begin`：若模型是 FSDP/FSDP2 且 `args.fsdp_config.activation_cpu_offload == true` 则启用；
- 同时读取 `fsdp_config.activation_checkpointing` 来协同 checkpoint 行为。

职责补充：

- 回调负责把“配置层面的开关”映射到代码执行（启用 offload、启用/替换 checkpoint、必要时 `enable_input_require_grads`）；
- 真正的 offload 逻辑完全在 handler/hook/forward wrap 内部，callback 只做一次性初始化。

## 价值与权衡（Value / Tradeoffs）

**价值**

- 通过把 saved activations 移到 CPU 降低 GPU 峰值显存；
- 在同样显存下支持更大 batch / 更长序列 / 更大模型；
- 异步 handler 试图让拷贝被计算覆盖，在 transformer 类工作负载上减少性能损失。

**权衡**

- 增加 CPU 内存占用（激活的 CPU 副本）；
- 增加 CPU↔GPU 带宽压力，PCIe 下更容易成为瓶颈；
- 引入同步/调度复杂性，性能高度依赖模型深度与 compute/transfer 比例。

## 技术难点与应对（Technical Challenges）

1) **autograd 正确性**
- 必须在 backward 需要时以正确 dtype/device/shape 恢复 saved tensors；
- saved-tensor hooks 能保证与 autograd 语义对齐。

2) **调度与同步**
- offload 过早/过晚都会影响性能与效果；
- `GroupCommitFunction` 提供 forward/backward 两侧确定的同步点；
- 异步 handler 用专用 stream + 显式 wait 保证执行顺序正确。

3) **避免 offload 错误对象**
- 通过 `FSDPParameterFilter` 用 storage 指针过滤参数相关张量。

4) **FSDP wrapper 兼容**
- 遍历并定位 FSDP/FSDP2 模块，wrap 其 underlying module 的 forward；
- 同时兼容 FSDP1（`FullyShardedDataParallel`）与 FSDP2（`FSDPModule`）。

5) **与 checkpoint 的交互**
- checkpoint 会改变哪些张量会被保存；
- 该实现提供 `ActivationHandler` 内部 checkpoint 路径，规避与 HF 原生实现的已知冲突。

## 如何启用（概览）

总体上通过训练配置启用：

- 使用 FSDP/FSDP2 训练策略 wrap 模型；
- 设置 `fsdp_config.activation_cpu_offload = true`；
- 可选设置 `fsdp_config.activation_checkpointing = true` 与 offload 联合使用。

精确行为以 `ActivationCpuOffloadCallBack.on_train_begin()` 与 `enable_activation_offloading()` 为准。

## 限制与后续改进（Limitations / Future work）

- “group≈layer” 属于启发式，复杂模型可能需要更模型感知的调度；
- 当前排除了 embedding-only 的 FSDP wrapper，其他小模块也可进一步过滤；
- 缺少对带宽、stall 时间、显存节省的可观测指标，后续可加 telemetry 以便调参。
