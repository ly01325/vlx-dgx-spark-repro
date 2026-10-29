# vlx-seek-1.5-3B 性能优化指导文件

> **版本** v2 · 2026-09-15 · 取代 v1(2026-09-14)
> **基线** NVFP4(仅语言 linear),E2E 1189 ms,DGX Spark GB10 ×1,TP=1,vLLM 0.19.0 定制分支
> **目标** 纯性能。精度门暂缓执行,保留于附录 A,不得删除。
> **范围** 已实现的 `OMCHAT_*` 环境开关**不作为交付项**;它们仅作测量仪器,见附录 C。

---

## 0. 本文件怎么用

按第 2 节的四个阶段推进。每个阶段的收益估计都标了置信度:

- **实测** —— 来自 `artifacts/results/` 或服务日志
- **推算** —— 从 FLOP/带宽算出,误差 ±30%
- **估计** —— 工程经验,误差可能 2×

**不要把不同阶段的收益相加。** 同一模块的优化相乘不相加,尤其是 P1 会改变 P4 所有条目的基准线。

---

## 1. 时间去哪了

### 1.1 模块拆解(实测,`results/reference/module_latency_nvfp4.json`)

| 模块 | ms | 占 E2E | 瓶颈类型 | 量化状态 |
|---|---|---|---|---|
| `shared_image_pack` | 5.4 | 0.5% | CPU | — |
| `main_mm_preprocess` | 17.9 | 1.5% | CPU | — |
| `main_vision_tower` (Qwen3.5 ViT, 24层 d=1024) | 282.6 | 23.8% | **compute** | BF16 未量化 |
| `aux_vision_tower` (C-RADIOv4-SO400M, 27层 d=1152) | 434.0 | **36.5%** | **compute** | BF16 未量化 |
| `roi_extract` (SimpleFPN + ROI + object 特征) | 70.6 | 5.9% | mixed | BF16 |
| `aux_projector` | 0.32 | 0.03% | — | BF16 |
| **视觉侧小计** | **788.3** | **66.3%** | | |
| LM prefill | 77.6 | 6.5% | compute | NVFP4 W4A4 |
| LM decode(见 1.2) | 269.2 | 22.6% | **bandwidth** | 部分 |
| 其余 | ~54 | 4.5% | | |
| **E2E** | **1189.3** | | | |

### 1.2 decode 的真实构成(关键,v1 未厘清)

服务日志实测:

```
NVFP4: decode_steps=26 total=269.4 model_step=134.1 other=135.3   artifacts/logs/vllm_server_nvfp4.log:7851
BF16 : decode_steps=26 total=531.6 model_step=406.2 other=125.3   artifacts/logs/vllm_server.log:7610
```

- `model_step/26 = 5.16 ms` —— 即 PERFORMANCE.md 里的 5.08 ms/token
- `other/26 = 5.20 ms` —— **profiler 完全没覆盖**。`compute_logits` 在 CUDA graph 和 profiler scope 之外执行

**真实 decode 步 ≈ 10.35 ms,其中约 4.15 ms 是 BF16 lm_head(1.0166 GB @ 245 GB/s ≈ GB10 峰值带宽 90%)。**

证据链:BF16 与 NVFP4 两次运行 `model_step` 差 3 倍,而 `other` 只差 0.4 ms/步 —— 说明 `other` 与语言层量化无关。⚠️ 这是推算,**P0 必须用 profiler scope 坐实**。

### 1.3 可攻击预算的硬边界

| 预算 | 值 | 说明 |
|---|---|---|
| 视觉侧 | 788 ms | 唯一存在 >30% E2E 空间的地方 |
| decode 侧全部工作 | **226 ms** | = E2E − TTFT = 1189.3 − 963。**不是 269 ms** —— 第一个 decode step 在 TTFT 内 |
| prefill | 77.6 ms | 激活位宽只影响这里 |

### 1.4 算力利用率(推算)

| 塔 | FLOP | ms | 达成 | GB10 BF16 峰值 |
|---|---|---|---|---|
| Aux RADIO | 15.3 TFLOP | 434 | **35 TFLOPS** | ~125 TFLOPS |
| Main ViT | 11.5 TFLOP | 283 | **41 TFLOPS** | ~125 TFLOPS |

**两塔都只跑到峰值的 28–33%。** 注意力占 aux 塔 MAC 的 54%(N=8170 全局,`mm_vitdet_window_size: null`),权重量化碰不到这一半。

**这 3× 的差距就是 P1 引擎化的理由。**

---

## 2. 优化策略:四阶段

| 阶段 | 内容 | 预期 E2E | 置信度 |
|---|---|---|---|
| **P0** | 测量与归属确认 | 1189(不变) | — |
| **P1** | 视觉塔引擎化(TensorRT) | ~925 | 估计 |
| **P2** | 精度 × kernel 后端择优 | ~865 | 估计 |
| **P3** | lm_head 在线 FP8 | −54 ms,可与 P1 并行 | 推算 |
| **P4** | 其余模型级优化 | 见 3.5 | 混合 |

**原则:先把每个模块放到它该在的引擎上(P1),再在引擎内部调优(P2)。** 顺序颠倒会导致 P2 的结论在 P1 之后全部作废。

---

## 3. 各阶段详述

### 3.0 P0 —— 测量与归属确认(前置,数天)

必须先做,否则后面所有投入都是赌。

| 动作 | 目的 | 产出 |
|---|---|---|
| 给 `compute_logits` 单独加 profiler scope | 坐实 135 ms `other` 桶的归属 | 决定 P3 值不值 96–114 ms |
| 视觉塔 kernel-level profile(nsys / torch profiler) | 434 ms 在 attention / GEMM / layernorm / elementwise / patch-embed 之间怎么分 | 决定 P1 的目标和 P2 的搜索空间 |
| 建立时延回归基线脚本 | 每次改动可比较 | 固定 harness |

⚠️ 现有 `bench_latency_throughput.py` 重放的是**同一张图、同一 prompt、同样 100 个 box、temp 0、max_tokens 27**。做时延回归够用,但任何涉及缓存或投机解码的测量在这个 harness 上都是假数据。

---

### 3.1 P1 —— 视觉塔引擎化(TensorRT)

**为什么适配**:两座塔都是**静态 shape、无状态、纯前馈**,没有 KV cache 也没有动态控制流 —— 理想 TRT 目标。28–33% 的算力利用率正是 TRT 的算子融合 + 针对具体 GPU 的 kernel autotune 该吃掉的部分。

**它还结构性地解决了一个已确认的问题**:整棵树里没有任何 GB10 的 autotune 配置。`vllm_/kernels/helion/configs/` 只有 `nvidia_h100.json` 和 `nvidia_h200.json`;`fused_moe/configs/` 里的 Blackwell 条目是 RTX PRO 6000,且本模型不是 MoE。

**先做 aux 塔**:更大(434 vs 283 ms),且比主塔干净 —— 主塔是 Qwen3.5 ViT,嵌在 vLLM 多模态路径里,耦合更深。

#### 最大的返工风险:中间层输出

ROI 提取消费的是**四个中间层的输出**,不是最终输出:

```python
# object_modules.py:746-747
self.out_indices = [round((i + 1) * num_layers / 4) - 1 for i in range(4)]
self.out_indices[-1] = num_layers - 1
```

**TRT engine 必须把这 4 个中间张量全部标记为 engine output。** 等引擎建完才发现,就得重建。

#### ✅ 边界决策已解决 —— 详见 `TRT_AUX_TOWER_SPEC.md`

**结论:引擎 = 27 层 block 栈,token-major 进出,ROI 留在 PyTorch wrapper 里。**

| 项 | 结论 |
|---|---|
| **部署下跑的是哪条路** | streamed ROI 路径(`_forward_rois_tensor` → `_forward_token_rois_eager`)。`_forward_intermediates` / legacy / pure_tensor / cudagraph 在单图部署下**全是死代码**。日志实证:`artifacts/logs/vllm_server.log:126` `Enabled streamed OmChat aux ROI extraction` |
| **四个 tap 层号** | **`[6, 13, 19, 26]`**(0-based),硬编码即可 |
| **引擎输入** | `tokens [B, L, 1152]` bf16,参考输入 `[1, 8170, 1152]`(10 前缀 + 68×120) |
| **引擎输出** | 4 × `[B, L, 1152]` bf16,**不剥前缀**(下游 `_roi_align_tokens` 自己剥) |
| **物化 4 张特征图的代价** | 75.2 MB 写 + 75.2 MB 读 ≈ **0.55 ms**,占 434 ms 的 **0.13%** —— 可忽略,所以切在 ROI 之前是对的 |
| **需要 plugin 吗** | **不需要**。roi_align 留在 PyTorch |
| **导出路线** | PyTorch → ONNX(fp32,opset≥17)→ TRT,`BuilderFlag.BF16` |

**唯一真阻塞:`AuxFlashAttention`**(不可 ONNX 导出)。导出脚本必须先设 `OMCHAT_AUX_FLASH_ATTN=0 OMCHAT_AUX_VARLEN_BATCH=0`,退回 timm 原生 `Attention`(走 SDPA)。

其他担心的导出阻塞**全部在切点 A 下消失**:CPE 位置编码、einops、antialias interpolate、InputConditioner 都在引擎外;spectral reparam **不存在**(`args.spectral_reparam=false`,checkpoint 无相关权重);feature normalizer **不存在**(两个 config 均 null);**无 forward hook**;27 层固定循环无数据相关控制流。

#### 🔴 最高返工风险:tap 层序/取点错误

层序错(把 binding index 当输出顺序 —— TRT **不保证**它与 ONNX 输出顺序一致)或 tap 取在 attn 之后而非 block 末尾,产生的张量**形状、dtype、数值范围全部正确**,不抛异常、不出 NaN,只是检测质量静默下降。而 `mm_projector_aux.0.weight [2048, 6656]` 的列布局把这个顺序硬编码了。

强制防御:① binding 一律**按名字**取;② 验证时必须**按 1152 一段切成 4 段分别算 cos**,绝不能只看整体 —— 层序错误的判别特征是"整体 cos 仍有 0.9x,但某一段远低于其它三段"。

#### 预期与工期(比原估计大幅下调)

**434 ms → 约 220–260 ms**(估算)。注意 434 ms 里已含 4 次 roi_align + mean pool(约 3–4 ms),引擎最多认领 ~430 ms。

⚠️ **434 ms 已经是 flash-attention 的数字**(`OMCHAT_AUX_FLASH_ATTN` 默认 True,日志 `artifacts/logs/vllm_server.log:39` 确认 27/27 已替换)—— 没有"上 flash attention"这块红利。

**工期:BF16 可用约 5.5–6 天,FP8 再加 2–3 天。** 原估计 2–4 周是高估的。

---

### 3.2 P2 —— 精度 × kernel 后端

#### 视觉侧:基本被 P1 吸收

TRT 本来就替你选 kernel。这一步退化成"分别用 BF16 / FP16 / FP8(以及 TRT 版本支持的话 NVFP4)建引擎,比时延" —— 搜索空间从几十个旋钮缩成 3–4 个 build 配置。

#### LM 侧:真正要比后端的地方

当前 `NvFp4LinearBackend.FLASHINFER_CUTLASS`(日志确认)。候选:

| 后端 | 状态 |
|---|---|
| FLASHINFER_CUTLASS | 现用,原生 Blackwell |
| Marlin | `modelopt.py:1089` 对 W4A16 强制走它。**sm_121 可用性待确认** |
| [vllm-project/humming](https://github.com/vllm-project/humming) | JIT 量化 GEMM,weight-only,不走 cuBLAS。**README 写 SM75+~Hopper,Blackwell(SM120/121)只在 FP4 激活那一行被提到 —— sm_121 + BF16 激活是否支持必须先验** |

**Humming 的两个前置检查**(各半天):
1. sm_121 + BF16 激活的 kernel 是否存在 —— 不存在则此路不通
2. 在 248182×2048、M=1 这个具体 shape 上实测有效带宽

> ⚠️ 反直觉点:现在的 BF16 lm_head 走 cuBLAS 纯流式读取能吃满 90% 带宽;带反量化的 kernel 因为要解包,M=1 时效率常常掉下来。已知 issue:*"INT3 Humming on SM80 achieves only ~43% effective HBM bandwidth at M=1"*。43% 效率下 INT4 仍然赢(字节少 4 倍),但收益从 81 ms 缩到 52 ms。**不要默认量化 kernel 一定更快。**

#### ❌ W4A8 在 GB10 上不可用

```
cutlass.py:37  "CUTLASS W4A8 requires compute capability of 90 (Hopper)"
cutlass.py:52  "CUTLASS W4A8, only supported int4"
```

`CutlassW4A8LinearKernel` 是 Hopper 专属(WGMMA + FP8),GB10 是 sm_121。而且它是 INT4 权重 + FP8 激活,与现有 modelopt/NVFP4 路线是两套东西。

**GB10 上的真实菜单只有:W4A4(现状)/ W4A16(weight-only)/ W8A8(FP8)。**

#### 激活位宽的取舍(若后续需要)

激活位宽**只影响 prefill,不影响 decode**(batch-1 时激活只有一个 2048 向量):

| 方案 | prefill | decode |
|---|---|---|
| W4A4(现状) | 77.6 ms | 基准 |
| W4A16(weight-only) | ~125 ms(+48) | **不变** |
| W8A8 FP8 | — | **更慢**(权重字节翻倍) |

**A4 只买了 ~48 ms 的 prefill,别的什么都没买。**

#### 🔴 一个未标定的风险

```
quantize_vlx_seek_native_nvfp4_w4a4.py:463   "size 16 and direct input scale 1.0."
quant_config.json                            activation_scale: {mode: "direct", value: null}
→ input_scale_calibrated: false
```

**186 层的全局激活缩放全部是 1.0,从未标定。** per-16 UE4M3 块缩放由 FlashRT 运行时算,是动态的;但全局缩放是单位值。这是 W4A4 最激进的配置。标定它是 186 个标量的改动,**零性能代价、零格式变更**,属于精度工作(附录 A · S1),当前阶段暂缓,但记录在案。

---

### 3.3 P3 —— lm_head 在线 FP8(约 1 天,可与 P1 并行)

**v1 对这一项的估计是错的,更正如下。**

树里已经实现了在线 FP8 embedding 量化:

```python
# fp8.py:330-335
elif isinstance(layer, VocabParallelEmbedding):
    if not self.quantize_embeddings:
        return None
    if self.is_checkpoint_fp8_serialized:
        return Fp8EmbeddingMethod(self)
    return Fp8OnlineEmbeddingMethod(self)       # ← 加载时量化,无需重导出
```

- `Fp8EmbeddingMethod(_Fp8EmbeddingMixin, Fp8WeightOnlyLinearMethod)`([:573](../../vllm_/model_executor/layers/quantization/fp8.py:573))= **weight-only FP8 = W8A16**,正是纯带宽层该用的档位
- dispatch 的是 **`VocabParallelEmbedding`** —— tie 时 `self.lm_head = self.model.embed_tokens` 恰好是这个类型,**`tie_word_embeddings=True` 不是障碍**
- `quantize_embeddings` 默认 False([:119](../../vllm_/model_executor/layers/quantization/fp8.py:119)),可经 `FP8_QUANTIZE_EMBEDDINGS` 环境变量([:189](../../vllm_/model_executor/layers/quantization/fp8.py:189))或 quant config 键([:254](../../vllm_/model_executor/layers/quantization/fp8.py:254))打开

**唯一缺口**:服务跑的是 `--quantization modelopt_fp4`,而 `modelopt.py` 里**没有任何 `VocabParallelEmbedding` / `ParallelLMHead` 分支**(grep 零命中)→ `get_quant_method` 对 lm_head 返回 None,静默走未量化。

**要做的事:把 `fp8.py:330-335` 那 6 行移植到 `modelopt.py` 的 `get_quant_method`。**

#### v1 vs 实际

| | v1 说的 | 实际 |
|---|---|---|
| 解绑 tie | 必须 | **不需要** |
| 重新导出 | 必须 | **不需要** |
| 代码改动 | 3 处 | **1 处** |
| 工作量 | 1–2 周 | **约 1 天** |
| 收益 | 54 ms | 54 ms(推算) |

**收益**:1.0166 GB → 0.508 GB,@245 GB/s 省 ~2.07 ms/步 × 26 步 ≈ **54 ms**。

想要 W4 而不是 W8 才是大工程 —— NVFP4 的 embedding 方法树里不存在,要自己写(预期 96–114 ms,但数周)。**先拿 FP8 这 54 ms。**

---

### 3.4 P4 —— 其余模型级优化

⚠️ **所有数字都是相对当前 434/283 ms 基线的。P1 落地后必须重测。**

| # | 动作 | 预期 | 成本 | 依据 |
|---|---|---|---|---|
| 1 | **辅助塔深度截断 27→20** | −113 ms | 改 config,数小时 | 见下 |
| 2 | **ViTDet 窗口注意力移植** | −170 ms | 1–2 周 | 注意力占 aux MAC 54% |
| 3 | **主塔 NVFP4 scope 扩展**(98 个 linear) | −40~70 ms | 纯导出脚本,数天 | 主塔 linear 本就是 vLLM 量化感知层 |
| 4 | **辅助塔 FP8 量化** | −35~75 ms | **远低于 v1 估计**,见下 | `AuxFp8Linear` 已存在 |
| 5 | **降低 multi-level ROI 取层点** | 乘数效应 | 中 | 见下 |
| 6 | **残余 per-step 开销** | −27 ms | 先 profile | P3 之后剩余部分 |

#### #1 深度截断 —— 纯配置,无需改代码

```python
# extra_timm_models.py:57-61
model_args = dict(patch_size=16, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=mlp_ratio)
... **dict(model_args, **kwargs)        # kwargs 覆盖 model_args
```

在 `C-RADIOv4-SO400M/config.json` 的 args 里加 `"model_kwargs": {"depth": 20}`。取层点自动适配(`num_layers = len(backbone.blocks)`,[object_modules.py:718](../../vllm_/model_executor/models/omchat/object_modules.py:718)),**一行代码不用改**。

成本随层数线性:27→20 = −26% × 434 = −113 ms;27→16 = −40% = −174 ms。

**建议直接扫 27/24/20/16/12,一次实验拿整条深度-时延曲线。**

待确认:权重加载路径对缺失层的容忍度(多余 block 权重会变成 unexpected keys)。

#### #2 窗口注意力 —— 因排除分辨率开关而增值

v1 把它折价到 ~40 ms,理由是"与分辨率 cap 重叠"。**现已排除分辨率开关,重叠不存在,它值回全额 ~170 ms。**

工作量真实:`vitdet.py` 在 `resources/.../C-RADIOv4-SO400M/` 里有,但**未被 vendored 进** `vllm_/model_executor/models/omchat/c_radio_v4/`。原版 `ViTDetHook` 注册在 `nn.Sequential` 的 forward hook 上,而本运行时三条路径全是**遍历 `backbone.blocks`** —— hook 永不触发,须重写接入方式。

🔴 **即使不管精度也要注意**:4 个 ROI 取层点中有 3 个会落在窗口化 block 上,产生**空间顺序错乱**的特征。这不是精度下降而是特征无意义,模型可能完全退化 —— 那样测到的时延也没有参考价值。移植时必须保证取层点落在全局注意力块上。

#### #4 辅助塔量化 —— v1 的判断是错的

v1 称"需要新建运行时 nn.Linear 替换类,3–5 周"。**该类已经存在:**

| 组件 | 位置 |
|---|---|
| `AuxFp8Linear` | [object_modules.py:229](../../vllm_/model_executor/models/omchat/object_modules.py:229) |
| `process_weights_after_loading` | [:275](../../vllm_/model_executor/models/omchat/object_modules.py:275) |
| `replace_aux_vision_linear_with_fp8` | [:474](../../vllm_/model_executor/models/omchat/object_modules.py:474) |
| `get_aux_vision_fp8_targets` | [:457](../../vllm_/model_executor/models/omchat/object_modules.py:457) —— **从 config 读,不是 env** |
| `FlashRTNvFp4TorchLinear` | [:348](../../vllm_/model_executor/models/omchat/object_modules.py:348) |

⚠️ 但存在**格式不匹配**:`FlashRTNvFp4TorchLinear` 要 `weight_packed` / `weight_scale` / `weight_global_scale`,而现有导出器写的是 `.weight` / `.input_scale` / `.weight_scale` / `.weight_scale_2`,且 [:396-404](../../vllm_/model_executor/models/omchat/object_modules.py:396) 在 `weight_global_scale != 1.0` 时直接抛异常。NVFP4 路径需要新导出分支;**FP8 路径(`AuxFp8Linear`)则相对干净**。

**覆盖范围已确认**:108/109 个 Linear —— 27×(`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`);**不覆盖** `patch_generator.embedder`(`ViTPatchLinear`, 768→1152)。无 adaptor。名字后缀匹配,FlashAttention 替换后仍能命中。

**为什么现在没生效**:`get_aux_vision_fp8_targets` 读的是 config 属性 **`aux_vision_quantization_config`**,要求 `quant_method == "fp8"`。该键在 `resources/.../config.json` 和 `C-RADIOv4-SO400M/config.json` 中**都不存在** → 返回空集 → 零替换。

#### ⚡ 一个 5 分钟的对照实验

在 `config.json` 里加:

```json
"aux_vision_quantization_config": {"quant_method": "fp8", "targets": ["mlp", "attn"]}
```

**不写任何新代码**,就能量到 aux 塔 FP8 weight-only 的收益上界 —— 用来给 P2 的 TRT FP8 阶段定预期。**建议在 P0 阶段就做掉。**

⚠️ 与 P1 的关系:TRT engine 的权重是构建期烘焙的,替换类会失效。两者互斥,这个实验只用于**定预期**,不是交付路径。

#### #5 取层点决定能截多深(乘数)

`out_indices[-1] = num_layers - 1` 强制最后一层必须被 tap → **永远要跑完整深度**。

`mm_aux_roi_extract_type: concat_multi_level_feat` 用 4 层 × 1152 = 4608 维。改成更少、更浅的取层点,**截断空间立刻放大** —— 这是让 #1 走得更远的乘数。属于改 object head 结构,比截深度重,比量化辅助塔轻。

---

## 4. 已排除的方案

| 方案 | 结论 | 依据 |
|---|---|---|
| **n-gram 投机解码** | ❌ 接受率结构性为零 | `omchat_qwen3_5_vl.py:323` 把后缀拼成 `"".join(f"<obj{i}><objfeat>")` → prompt 里 `<objN>` 后永远是 `<objfeat>`,而 `<objfeat>` 从不被生成,每次提案必被拒 |
| W4A8 | ❌ GB10 不可用 | `cutlass.py:37` Hopper 专属 |
| KV cache 量化 | ❌ 只占 decode 流量 3.4% | 26.6 MiB vs lm_head 969.5 MiB;且会换掉最快的 FLASH_ATTN 后端 |
| 换小号 C-RADIO 变体 | ❌ 只省 1.3× FLOP 却要重训整条 object 路径 | `mm_projector_aux.0.weight` 的 `[2048, 6656]` 写死 |
| 改 head_dim / 换注意力后端 | ❌ FA2 支持 head_dim 256,全注意力层不是瓶颈 | |
| 移除 `attn_output_gate` | ❌ 已在量化 scope 内,成本仅 ~3 ms | 186 → 114 融合 = 18×5 + 6×4 |
| 原样移植 `ViTDetHook` | ❌ hook 永不触发 | 见 3.4 #2 |
| 词表裁剪 lm_head | ❌ 换 2% E2E 却永久封闭输出字母表 | `vllm_request.py:140-142` prompt 自由文本;`mm_use_think_token: true` |

### 关于投机解码的补充

**混合线性注意力不阻塞投机解码** —— 这一点 v1 的担心是多余的。`gdn_attn.py` 的 `num_accepted_tokens`(:60/:142/:301)配合 `spec_state_indices_tensor` 实现完整,`qwen3_5.py:750-762` 按 `num_speculative_tokens` 给 GDN state 扩容。

但:
- 可攻击预算只有 **226 ms**,γ=3 下盈亏平衡接受长度 **a = 1.56**,n-gram 达不到
- **suffix decoding** 才是对的提案器(匹配历史响应而非 prompt),但依赖 `arctic_inference`:aarch64 轮子存疑,且该包注册 `vllm.general_plugins` 入口点会被 `vllm_/plugins/__init__.py:69` 自动加载并 monkey-patch 本定制分支 —— 真实冲突风险
- **MTP**:`text_config.mtp_num_hidden_layers: 1` 存在,`qwen3_5_mtp.py` 运行时完整,但 checkpoint 里 **0 个 `mtp.*` 张量**,要从头蒸馏,数周
- **Medusa**:`medusa.py:46` 只支持 top-1 链式提案、无树注意力

**结论:投机解码整体优先级低于 P1–P3,且必须在 lm_head 量化之后重新评估**(量化后每步从 10.35 → ~7 ms,盈亏平衡点抬高,绝对收益缩水约 40%)。

---

## 5. 路线图

从 1189 ms 起。**收益不可简单相加。**

| 阶段 | 动作 | 预期 E2E | 累计 | 工作量 | 置信度 |
|---|---|---|---|---|---|
| P0 | profiler scope + kernel profile + 回归基线 + aux FP8 对照实验 | 1189 | 0 | 数天 | — |
| P3 | lm_head 在线 FP8(与 P1 并行) | ~1135 | −4.5% | **1 天** | 推算 |
| P1a | aux 塔 TRT engine(BF16 跑通) | ~960 | −19% | **5.5–6 天** | 估计 |
| P1b | main 塔 TRT engine | ~880 | −26% | 1–2 周 | 估计 |
| P2 | FP8 引擎重建 + LM kernel 后端择优 | ~820 | −31% | 1–2 周 | 估计 |
| P4-1 | aux 深度截断(扫曲线) | ~790 | −34% | 数小时 | 推算 |
| P4-3 | 主塔 NVFP4 scope 扩展 | ~750 | −37% | 数天 | 推算 |
| P4-2 | 窗口注意力移植 | ~680 | −43% | 1–2 周 | 估计 |

**现实落点:P0+P3+P1 ≈ 925 ms(−22%),约 1–2 个月。** 这部分不依赖任何现有开关,也不依赖精度评测。

再往下(P2/P4)误差棒明显变大,且 P4 的数字必须在 P1 之后重测。

---

## 6. 待确认清单

| # | 问题 | 怎么确认 | 阻塞什么 |
|---|---|---|---|
| 1 | 135 ms `other` 桶是否真是 lm_head | 给 `compute_logits` 加 profiler scope | P3 的收益大小 |
| 2 | 434 ms 在 attention/GEMM/其他之间的分布 | nsys 或 torch profiler | P1 目标、P2 搜索空间 |
| 3 | ~~TRT 边界切在 ROI 前还是后~~ | ✅ **已解决** —— 切 ROI 之前,见 `TRT_AUX_TOWER_SPEC.md` §0 | — |
| 4 | Humming 是否支持 sm_121 + BF16 激活 | 读其 dispatch 表 / 编译目标 | P2 的 LM 侧选项 |
| 5 | Marlin W4A16 在 sm_121 可用性 | `modelopt.py:1089` 路径实测 | P2 备选 |
| 6 | 权重加载对 aux 缺失层的容忍度 | 截断深度后启动一次 | P4-1 |
| 7 | ~~`AuxFp8Linear` 的覆盖范围与 config 键~~ | ✅ **已解决** —— 108/109 个 Linear,缺 `aux_vision_quantization_config` 键 | — |
| 8 | ~~中间层收集是否用 forward hook~~ | ✅ **已解决** —— 无 hook,切点 A 不经过 `forward_intermediates` | — |
| 9 | 本机 TensorRT 版本是否支持 sm_121 | `python3 -c "import tensorrt; print(tensorrt.__version__)"` | **P1 开工** |
| 10 | LayerNorm eps / GELU 变体 | 见 `TRT_AUX_TOWER_SPEC.md` §7 探针表(共 8 项) | 仅手搭 TRT 路线阻塞 |

> 📌 `TRT_AUX_TOWER_SPEC.md` §7 给出了 8 个上机探针的完整命令和期望值,**P1 步骤 0 就是把它们跑完**(0.5 天)。

---

## 附录 A · 精度验证方案(暂缓执行,不得删除)

当前阶段不设精度门。但以下事实必须在**任何交付前**处理:

### A.1 本包无法用于精度评测

`tensor_provenance.json` 显示 **46 个 `fresh_initialization` 张量恰好是整个 object head**:`mm_projector_aux.*`(4 个)+ `object_vp_extractor.simple_fpn.*`(42 个)。

**这份 sanitized checkpoint 的物体头是随机初始化的。** 在它上面测出的任何索引/标签准确率都无意义。330 个 aux 张量是 stock `official_radio`,LM 是 `official_qwen` —— 本包不含 VLX 私有微调权重。

`tokenizer.json` / vocab / merges 也未同步(只有 `tokenizer_config.json`)。

### A.2 仓库内现无任何精度评测

`verify_export.py` / `smoke_load.py` 只做结构校验(张量名、dtype、scale 形状);`bench_*.py` 是纯时延。**没有任何一处比对过输出内容。**

### A.3 三层评测(交付前必须建立)

| 层 | 内容 | 成本 | 可在本包执行? |
|---|---|---|---|
| **T1** | 离线权重反量化 vs 源 BF16 的余弦 / 相对 Frobenius 误差,语言层 cos > 0.999 | 秒级,无需 GPU | ✅ 可以,**今天就能加** |
| **T2** | 逐 token 等价性:量化 vs BF16,≥30 组固定输入,temp=0,要求逐 token id 完全一致 | 分钟级 | ✅ 自洽性检查,不需要真实 object head |
| **T3** | 检测精度:≥500 张留出图,序列精确匹配率 / 逐物体 P-R / 标签正确率 / 格式畸形率 | ~1 GPU-小时 | ❌ **需私有 checkpoint** |

**T3 必须分层**,不分层等于没测:
- 按框面积(COCO small <32² / medium / large)—— 分辨率类改动的失效集中在小框。数学断点明确:主塔 1408 宽下,原宽 ~75 px 的框在 FPN p2 上恰好映射成 14 px = `mm_roi_output_size`,低于此开始插值
- 按类别频次 —— 深度截断/蒸馏的失效集中在稀有类
- 按"100 候选框中同类候选数" —— 本任务失效模式是**判别**而非定位

### A.4 各阶段的精度门(恢复精度门后启用)

| 阶段 | 门槛 |
|---|---|
| 激活 scale 标定(S1) | T1 全通过;T3 不劣于当前 NVFP4 基线 |
| 主塔量化 | T1 cos>0.999;T3 序列精确匹配率 delta > −0.5% |
| **lm_head 量化** | **T2 必须 100% 逐 token 一致**;不一致则降格重测 |
| 分辨率 / 深度截断 | 小框桶精确匹配率 delta > −1%,中大框在噪声内 |
| 辅助塔量化 / 手术 | 逐级 aux ROI 向量余弦 ≥0.98 作诊断,**发布门用 T3 端到端** |

---

## 附录 B · 关键代码位置索引

### 模型结构

```
resources/vlx-seek-1.5-3b-sanitized/config.json
  architectures            OmChatQwen3_5ForCausalLM
  model_type               omchat_qwen3_5
  mm_vision_tower          Qwen3.5-2B-Vision_Tower        ← 视觉塔,不是基座 LM
  mm_vision_tower_aux      C-RADIOv4-SO400M
  text_config.model_type   qwen3_5_text                    ← 基座 LM
  text_config.vocab_size   248182
  tie_word_embeddings      true(顶层与 text_config 均为)
  num_hidden_layers        24 = 18 linear_attention + 6 full_attention
  full_attention_interval  4
  hidden_size 2048 / head_dim 256 / 8 heads / 2 KV heads
  mm_num_object_tokens     100
  mm_object_hidden_size    6656 = 4×1152(aux) + 4×512(vt)
  mm_vitdet_window_size    null                            ← 全局注意力
```

LM 的线性注意力是 **Gated DeltaNet**(Qwen3-Next 同源):
- `qwen3_5.py:118` `Qwen3_5DecoderLayer(Qwen3NextDecoderLayer)`
- `:135-136` `if layer_type == "linear_attention": self.linear_attn = GatedDeltaNetAttention(...)`
- `:599` `Qwen3_5ForConditionalGeneration(..., IsHybrid)`
- 实现在 `layers/mamba/gdn_linear_attn.py`

### 量化

```
quant/quant_config.json                group_size 16, expected_count 186,
                                       activation_scale {mode: "direct", value: null}
quant/export_report.json               input_scale 186, uint8_weights 186, bf16_vision_object_weights 339
                                       output SHA256 6cdd01d4...ecb2acb1
quantize_vlx_seek_native_nvfp4_w4a4.py:34-35   DEFAULT_EXPECTED_COUNT 186 / FUSED 114
                                :46-53   FORBIDDEN_TARGET_SUBSTRS(vision_tower, radio_model, .visual., ...)
                                :169-179 硬性要求 startswith("model.language_model.layers.")
                                :463     "size 16 and direct input scale 1.0"
                                :563-573 _validate_export 断言 vision/object 权重仍是 BF16
```

⚠️ **运行时权威量化配置来自 `config.json['quantization_config']`(`weight_utils.py:296-307`),从不打开 `hf_quant_config.json`。两个文件必须同步改。**

### 视觉塔

见 3.1 / 3.4 内的引用。核心:`object_modules.py`(1538 行)的 `OmChatVisionTowerAux`(:711)。

---

## 附录 C · 现有 OMCHAT_* 开关(仅作测量仪器,非交付项)

**按项目决定,这些不作为优化交付项。** 但它们是最快的测量工具,可用于确定各条路线的天花板。默认值全部经 `_env_flag` / `_env_int` 核实:

| 开关 | 默认 | 可测量什么 |
|---|---|---|
| `OMCHAT_SKIP_AUX_VIT` | off | 辅助塔的 434 ms 上界 |
| `OMCHAT_AUX_MAX_PIXELS` | 0(不限) | 分辨率-时延曲线(**面积** cap,非长边) |
| `OMCHAT_PARALLEL_AUX_VIT` | off | 双塔重叠实际收益(独立 CUDA stream,`omchat_qwen3_5_vl.py:1297/:1349`) |
| `OMCHAT_AUX_VIT_CUDAGRAPH` | off | 图捕获收益 |
| `OMCHAT_COMPILE_AUX_VIT` | off | torch.compile 收益 |
| `OMCHAT_PARALLEL_VT_FPN` | off | FPN 并行 |
| `OMCHAT_STREAM_AUX_ROI` | **on** | 已启用 |
| `OMCHAT_BATCH_ROI_EXTRACT` | **on** | 已启用 |

⚠️ **记账提醒**:开关那套与本文件的方案是**互补而非重叠**的。双塔并行(可能 −283 ms)与 P1–P4 任何一项都不冲突。排除它们作为交付项时,不要在记账上把它们当成"已被覆盖"。

`aux_image_size: 1024` 是**死配置** —— `grep -rn aux_image_size vllm_/` 零命中,实际分辨率由 `do_resize=False` + 16 倍数对齐决定,即原生 1920×1088。
