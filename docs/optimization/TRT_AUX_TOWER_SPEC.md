# C-RADIOv4-SO400M 辅助视觉塔 TensorRT 集成接口规范

> 生成于 2026-09-15 | 配套 OPTIMIZATION_PLAN.md 的 P1 阶段
> 方法: 7 维并行代码追踪 -> 对抗式验证 -> 规格综合 (28 agents)
> ⚠️ 标 **未验证** 的条目必须上机确认后方可依赖

# C-RADIOv4-SO400M 辅助视觉塔 TensorRT 集成接口规范

适用：DGX Spark / GB10 (sm_121, aarch64), TP=1, vLLM 0.19.0 vendored @ `/Users/liuy/Projects/VLX/vlx_seek_1_5_3b/vllm_`
参考输入：1 张图，原生 1920(W)×1088(H)，100 个 bbox，batch=1
所有行号均为绝对路径下的真实行号。标 **未验证** 的条目必须在上机后确认。

---

## 0. 边界决策

### 0.1 现状：部署配置下跑的是哪条路

这是全篇最关键的前提。部署配置（仅开 profiling flag）下，塔**不返回四张特征图**，而是返回已经 ROI-pool 完的向量。

判定链（全部静态可判 + 日志实证）：

| 判据 | 值 | 出处 |
|---|---|---|
| `OMCHAT_STREAM_AUX_ROI` | 默认 **True** | `omchat_qwen3_5_vl.py:1381` (`_env_flag(name, True)`) |
| `mm_use_simpleFPN_for_vt_aux` / `_multiLayer_` | false / false | `resources/vlx-seek-1.5-3b-sanitized/config.json:47,49` |
| `mm_pos_embedding_strategy` | `"bbox_based"`（非 feature_map_based/hybrid） | `config.json:27` |
| `OMCHAT_COMPILE_AUX_VIT` / `OMCHAT_AUX_VIT_CUDAGRAPH` | 默认 False | `object_modules.py:54-58` |
| ⇒ `_can_stream_aux_rois()` | **True** | `omchat_qwen3_5_vl.py:1378-1395` |
| 运行时实证 | `artifacts/logs/vllm_server.log:126`、`artifacts/logs/vllm_server_nvfp4.log:131`：`[object_modules.py:993] Enabled streamed OmChat aux ROI extraction for tensor batches.` | — |

实际调用链（单图）：
```
omchat_qwen3_5_vl.py:1917  _run_aux_vision_tower(single_pixel_values_aux, [boxes])
  -> :1406-1412  vision_tower_aux(images, roi_boxes=..., roi_output_size=14,
                                  roi_spatial_scale=1/16, roi_aligned=True)
  -> object_modules.py:1502  forward(): roi_boxes is not None and not isinstance(images, list)
  -> :982   _forward_rois_tensor
  -> :926   _forward_token_rois_eager     # roi_align 融合在 27 层循环里，每个 tap 做一次
  -> 返回 dict{roi_features, roi_feature_counts, input_size, feature_map_size}
```
`_forward_intermediates`(:1438)、`_forward_intermediates_legacy`(:825)、`_forward_intermediates_pure_tensor`(:1151)、cudagraph 路径在单图部署下**全部是死代码**。
434 ms 的 profile 标签包住的是 `_run_aux_vision_tower`，因此**包含这 4 次 roi_align + mean pool**。

### 0.2 ROI 融合问题：两个选项

TRT 不能做 `torchvision.ops.roi_align`（无对应算子，需自写 plugin）。所以引擎边界必须切在 ROI 之前，四个中间层必须物化。两个选项：

**选项 A（推荐）— 引擎出 4 个中间层，wrapper 内部自己做 ROI，`forward()` 返回值与今天逐字节同构**
把 TRT 塞进 27 层 block 循环的位置，wrapper 仍旧返回 streamed dict。下游 `omchat_qwen3_5_vl.py` / `multilevel_roi_visual_prompt.py` **一行不改**。

**选项 B — 设 `OMCHAT_STREAM_AUX_ROI=0`，让塔回到 dense `image_features` 分支**
`aux_roi_features=None`，roi_align 在 `multilevel_roi_visual_prompt.py:685-697` 做。该分支下游完全支持（`omchat_qwen3_5_vl.py:1544-1580`），数值等价（四层同为 68×120，roi_align 逐通道，先 cat 后 align 与先 align 后 cat 等价）。但改变了运行分支，且多一次 `cat` 到 `[1,4608,68,120]`（+300 MB fp32 临时）。

### 0.3 物化四张特征图的带宽代价（实算）

| 张量 | 元素数 | bf16 | fp32 |
|---|---|---|---|
| 单层 `[1,1152,68,120]` | 9,400,320 | 18.80 MB | 37.60 MB |
| 四层合计 | 37,601,280 | **75.20 MB** | 150.41 MB |
| 单层 token 形 `[1,8170,1152]` | 9,411,840 | 18.82 MB | — |
| streamed 输出 `[1,100,4608]` fp32 | 460,800 | — | 1.84 MB |

关键：**今天的 streamed 路径已经在每个 tap 做了同样的 `reshape→permute→contiguous→.float()`**（`object_modules.py:912-919`）。选项 A 相对今天的**唯一增量**是引擎把 4 个 tap 写回 HBM、wrapper 再读回：75.20 MB 写 + 75.20 MB 读 ≈ **150 MB**。GB10 LPDDR5X ≈ 273 GB/s（规格值，**未验证** 实测带宽）⇒ **≈ 0.55 ms**，占 434 ms 的 0.13%。

**结论：ROI 融合带来的收益在带宽上可以忽略。采用选项 A。**

### 0.4 三个候选切点与推荐

| 切点 | 引擎输入 | 引擎输出 | 评价 |
|---|---|---|---|
| **A. tokens → 4 taps（推荐）** | `[1,8170,1152]` bf16 | 4 × `[1,8170,1152]` bf16 | CPE 位置编码、Im2Patches、einops、conditioner 全在宿主，引擎内只剩纯 ViT block 栈。导出最简单，动态维只有 L 一维 |
| B. tokens → 4×NCHW | 同上 | 4 × `[1,1152,68,120]` | 省一次 permute-copy（约 0.28 ms），但引擎内引入依赖 (H,W) 的 reshape，动态形状时需要两个 shape 输入。不值 |
| C. 图像入（full tower） | `[1,3,1088,1920]` fp32 | 同上 | 需在引擎内复刻 `F.interpolate(..., antialias=True)`（TRT `IResizeLayer` **无 antialias**，必然数值不符）+ CPE 位置编码的 `interpolate→crop`（是 (H,W) 的非线性函数）。profile 变成 H×W 笛卡尔盒，按 (H_max,W_max) 角点分配 workspace。**不推荐** |

**最终推荐：选项 A + 切点 A。** 引擎 = 27 层 block 栈，输入输出都是 token-major，**不在引擎内剥前缀 token**（见 §2.4）。

---

## 1. 引擎输入契约

### 1.1 宿主侧保留的预处理（全部不进引擎）

顺序固定，全部复用现有代码，不要重写：

```
pixel_values_aux                      uint8 [1,3,1088,1920] CUDA, NCHW contiguous
  -> _prepare_image                   object_modules.py:1165-1174
       .to(device=backbone.device, dtype=float32); if uint8: mul_(1/255)
       -> _resize_to_supported        :1464-1483
          min_resolution_step = patch_size = 16 (window_size=None，因 mm_vitdet_window_size=null,
          radio_model.py:110-114, object_modules.py:1531)
          1088 与 1920 均为 16 的倍数 => **恒等，直接 return image**
          (1920×1080 的原图会在这里被 bilinear+antialias 拉伸到 1088；round(1080/16)=round(67.5)=68)
  -> _prepare_token_inputs            :835-849
       conditioned = radio.input_conditioner(image)   # (x-mean)/std，fp32 计算后 .to(bf16)
       tokens = backbone.patch_generator(conditioned) # Im2Patches(16) -> ViTPatchLinear(768->1152,
                                                      # 无 bias) -> +pos_embed -> ClsToken 前置拼接
       返回 tokens, grid=(1088//16, 1920//16)=(68,120)
```

InputConditioner 常量（`input_conditioner.py:28-35`，`object_modules.py:797-803`）：buffer 声明为 fp32，但 checkpoint 里 `input_conditioner.norm_mean/.norm_std` 的 dtype 是 **BF16**（`tensor_provenance.json`，shape `[3,1,1]`），且持久 buffer 会被 `AutoWeightsLoader` 加载（`utils.py:231-243` 显式把 persistent buffer 并入可加载张量）。因此运行时生效的是 **bf16 舍入后的 CLIP 常量**，不是教科书 fp32 值：

| 通道 | mean (bf16, 精确十进制) | std (bf16, 精确十进制) |
|---|---|---|
| R | 0.482421875 | 0.26953125 |
| G | 0.45703125  | 0.26171875 |
| B | 0.408203125 | 0.275390625 |

若将 conditioner 挪进引擎，必须烧这六个值，不是 0.48145466/… （**未验证**：需上机 print 确认 buffer 实际值，探针见 §7）。

CPE 位置编码（`vit_patch_generator.py:194-267`，eval 分支）：`pos_embed [1,16384,1152]` → reshape 成 `[1,1152,128,128]` → `F.interpolate` 到 `(max(68,120), max(68,120))=(120,120)`（fp32 计算后转回 bf16）→ `window_select` 裁行到 68（列已是 120 不裁）→ flatten → `[1,8160,1152]`。**这是 (H,W) 的非线性函数**，是切点 C 被否掉的主因。切点 A 下它完全在宿主。

### 1.2 引擎输入张量

| 名称 | 形状 | 符号维解析（参考输入） | dtype | 布局 |
|---|---|---|---|---|
| `tokens` | `[B, L, 1152]` | B=1, L=8170 | **bfloat16** | token-major，contiguous，strides `(9411840, 1152, 1)` |

- `L = num_skip + (H/16)*(W/16)`，`num_skip = 10`。参考输入：`10 + 68*120 = 8170`。
- 前 10 个是前缀 token（3 CLS + 7 register，见 §2.4），**前置**拼接（`cls_token.py:49-52` `torch.cat([token, x], dim=1)`）。
- token 内空间顺序 **行优先**：`index = 10 + row*120 + col`，row∈[0,68), col∈[0,120)（`vit_patch_generator.py:283-286` 的 `rearrange('b c (py yy) (px xx) -> b (py px) (c yy xx)')`，py=68 在外）。
- 值域：无归一化约束，就是 patch embedding + pos embed 之后的激活。
- dtype 来源：`omchat_qwen3_5_vl.py:840-843` 把整个 aux 塔 `.to(dtype=self.visual 参数 dtype)`，`config.json` 顶层与 `vision_config` 均为 `"dtype": "bfloat16"`。C-RADIO 自带的 `args.dtype="float32"` 被这一步覆盖。

### 1.3 TRT optimization profile

只有 `L` 是动态维（B 见下）。约束：`H % 16 == 0 && W % 16 == 0 && H,W <= 2048`（`max_resolution=2048`, `cpe_max_size=2048`）。

**阶段一（强烈建议先做）：完全冻结**

| profile | 维 | MIN | OPT | MAX |
|---|---|---|---|---|
| `p_fixed` | B | 1 | 1 | 1 |
| | L | 8170 | 8170 | 8170 |

**阶段二（多分辨率）：三个 profile，避免单 profile 在极值被调坏**

| profile | L MIN | L OPT | L MAX | 覆盖 |
|---|---|---|---|---|
| `p_small` | 266 | 2058 | 8169 | ≤ 1 MPix（256×256 … 小图） |
| `p_main` | 8170 | 8170 | 8170 | 1920×1088 主力 |
| `p_large` | 8171 | 12298 | 16394 | 至 2048×2048（10+128×128） |

B 的处理：`MultiModalFieldConfig.batched("image")`（`omchat_qwen3_5_vl.py:670`）会把同 (device,dtype,shape) 的多图 stack 成 `[N,3,H,W]`，`_forward_rois_tensor` 直接按 `image.shape[0]` 跑（`:1003-1007`），所以 **B>1 在 streamed 路径上是可达的**。阶段一 B 固定为 1，wrapper 必须在 `image.shape[0] > 1` 时**逐图循环调引擎**（见 §3.4）。

风险提示：L=16394 时若 TRT 未选中 fused MHA，attention score 矩阵 `16 heads × 16394² × 2B = 8.6 GB`，会 OOM 或 workspace 爆炸。构建后必须用 `trtexec --verbose` 或 `IEngineInspector` 确认走的是 fused/flash MHA kernel。

---

## 2. 引擎输出契约

### 2.1 四个 tap 的层号（depth=27，已由 checkpoint 交叉验证）

```python
# object_modules.py:746-748
self.out_indices = [round((i + 1) * num_layers / 4) - 1 for i in range(4)]
self.out_indices[-1] = num_layers - 1
self._out_indices_tuple = tuple(self.out_indices)
```
`num_layers = len(backbone.blocks) = 27`（`object_modules.py:717-718`；depth 来源 `c_radio_v4/extra_timm_models.py:57-59` `depth=27`；checkpoint 实证：`tensor_provenance.json` 中 `...radio_model.model.blocks.0 … blocks.26`，恰好 27 个，无 27 及以上）。

⇒ **`[6, 13, 19, 26]`（0-based，升序）。硬编码即可。**

该公式在 depth=27 上**与舍入模式无关**：6.75→7、13.5→14、20.25→20、27.0→27（再 −1），round-half-even 与 round-half-up 结果完全一致。C++/Python/NumPy 任何重实现都不会分歧，无需为 tie 做特殊处理。

升序在每条路径上都被强制：token 路径按位置解包（`object_modules.py:856`、`:1096` `index0,index1,index2,index3 = self._out_indices_tuple`）；streamed ROI 路径用单调游标并在未消费完四个时抛异常（`:957-974`）；legacy 路径显式排序（`c_radio_v4/forward_intermediates.py:73` `take_indices = sorted(take_indices)`）。

### 2.2 tap 点的精确语义

tap 取的是 **block 完整输出（两次残差加之后）**，不是 post-attn，不是 pre-LayerScale。参考 `object_modules.py:949-957`：

```python
attention_output = attention(norm1(hidden_states))
hidden_states = hidden_states + drop_path1(layer_scale1(attention_output))
hidden_states = hidden_states + drop_path2(layer_scale2(mlp(norm2(hidden_states))))
# <-- tap 在这里
```
本 checkpoint 中 `ls1/ls2` 与 `drop_path1/2` 均为 `nn.Identity`（checkpoint 里不存在任何 `blocks.N.ls1.gamma`；`args.drop_path = null`；推理期 drop_path 恒等），`attn.q_norm/k_norm/attn.norm` 同样是 Identity（每层只有 12 个张量，见 §2.5）。因此每个 block 精确等于：

```
x = x + attn(LayerNorm(x))
x = x + mlp(LayerNorm(x))
```

### 2.3 无任何归一化，四条独立证据

1. `args.model_norm = false` ⇒ `radio_model.py:342-343` 把 `model.norm = nn.Identity()`；checkpoint 里没有 `...model.norm.weight/bias`。
2. `forward_intermediates` 以 `norm=False` 调用（`object_modules.py:830`）⇒ `enable_cpe_support.py:38` 解析成 `lambda y: y`。
3. `inter_feature_normalizer_config: null` 且 `feature_normalizer_config: null`（`C-RADIOv4-SO400M/config.json:270-271`）⇒ `hf_model.py:118-129` 两者均为 None ⇒ `forward_intermediates.py:83-84` 用 `NullIntermediateFeatureNormalizer`，`alpha` 是值为 1 的 0 维张量（`feature_normalizer.py:100,110-111`），`alpha * y` 逐位恒等。
4. `aggregation` 默认 `"sparse"`（`forward_intermediates.py:42`，`radio_model.py:257`），调用方未覆盖。**若误用 `"dense"`，得到的是每个四分组内所有 block 输出的均值——形状/dtype 完全相同的错误张量，这是本项目最隐蔽的失败模式。**

另：`args.spectral_reparam = false`，checkpoint 里每个 Linear 只有 `weight`/`bias`，**无 spectral reparam 权重，无需 disable 步骤**。
`_prepare_static_intermediate_normalizer`（`object_modules.py:1239-1247`）只是 CUDA-graph 捕获期的占位，数学上恒等，且只在 `_capture_aux_cudagraph`(:1263) 里被调用，部署下不执行——**完全忽略，不要在引擎里实现任何 alpha 缩放**。

### 2.4 输出张量表（推荐形态：不剥前缀、token-major）

| binding 名 | 来源 | 形状 | 参考输入解析 | dtype | 布局 |
|---|---|---|---|---|---|
| `feat_l06` | blocks[6] 完整输出 | `[B, L, 1152]` | `[1, 8170, 1152]` | bfloat16 | token-major contiguous |
| `feat_l13` | blocks[13] | `[B, L, 1152]` | `[1, 8170, 1152]` | bfloat16 | 同上 |
| `feat_l19` | blocks[19] | `[B, L, 1152]` | `[1, 8170, 1152]` | bfloat16 | 同上 |
| `feat_l26` | blocks[26]（最后一层，**无 final norm**） | `[B, L, 1152]` | `[1, 8170, 1152]` | bfloat16 | 同上 |

**为什么不在引擎里剥前缀**：宿主侧的消费者 `_roi_align_tokens`（`object_modules.py:901-925`）自己做 `hidden_states[:, prefix_tokens:]`。引擎输出保留 8170，wrapper 就能**逐字不改地复用 `_roi_align_tokens`**。若你决定在引擎里剥成 8160，必须在代码里显式标注，且 wrapper 必须把 `prefix_tokens=0` 传进去——重复剥会让 `reshape(1,68,120,-1)` 在 8150 个 token 上失败（至少是响亮的报错），而漏剥会让 `reshape` 在 9,411,840 元素上失败（不能被 8160 整除）。

**前缀 token = 10，checkpoint 实证**：`patch_generator.cls_token.token` 形状 `[10, 1152]`。构成 = `num_cls_tokens=3`（3 个唯一 teacher：siglip2-g / dino_v3_7b / sam3，`cls_token_per_teacher: true`，`radio_model.py:348-352`）+ `num_registers = register_multiple - (num_tokens % register_multiple) = 10 - 3 = 7`（`cls_token.py:30-31`，`register_multiple: 10`，`cpe_num_registers: null`）。`num_skip = 3+7 = 10`（`vit_patch_generator.py:119-121`），`self._prefix_tokens = backbone.patch_generator.num_skip`（`object_modules.py:763`）。索引 0-2 是 CLS，3-9 是 register。**下游不消费任何前缀 token。**

**输出顺序是 binding 名决定的，不要依赖 binding index**：TensorRT 不保证 binding 顺序与 ONNX 输出顺序一致。wrapper 必须按名字取，并显式按 `[feat_l06, feat_l13, feat_l19, feat_l26]` 组 list。

### 2.5 单层 block 的权重清单（每层 12 个张量，全 BF16）

| 名称 | 形状 |
|---|---|
| `norm1.weight` / `norm1.bias` | `[1152]` / `[1152]` |
| `attn.qkv.weight` / `.bias` | `[3456, 1152]` / `[3456]` |
| `attn.proj.weight` / `.bias` | `[1152, 1152]` / `[1152]` |
| `norm2.weight` / `norm2.bias` | `[1152]` / `[1152]` |
| `mlp.fc1.weight` / `.bias` | `[4304, 1152]` / `[4304]` |
| `mlp.fc2.weight` / `.bias` | `[1152, 4304]` / `[1152]` |

12 × 27 = 324，加 `patch_generator.{cls_token.token[10,1152], embedder.weight[1152,768], pos_embed[1,16384,1152]}`、`input_conditioner.{norm_mean,norm_std}[3,1,1]`、`radio_model.summary_idxs[2]`(I64) = **330 个张量，431,237,240 参数，BF16 共 862.5 MB**（其中 blocks 部分 411,466,608 参数 = 822.9 MB）。

超参：`num_heads=16`，`head_dim=72`，`scale = 72**-0.5 = 0.1178511301977579`，MLP hidden = 4304（`mlp_ratio = 4304/1152`）。
LayerNorm `eps` 与 GELU 变体（erf vs tanh）由 timm 默认决定，本仓库未固定（`extra_timm_models.py:57-59` 只传 patch_size/embed_dim/depth/num_heads/mlp_ratio，`model_kwargs={}`）。timm 当前默认是 `eps=1e-6` 与 `nn.GELU`（精确 erf），但 **未验证**——这正是推荐走 ONNX 导出而非手搭网络的理由（导出会自动捕获）。探针见 §7。

### 2.6 引擎之后消费者做什么（wrapper 必须复刻，逐行对应现有代码）

```python
# 每个 tap，object_modules.py:901-925 (_roi_align_tokens) —— 原样复用
spatial = feat[:, 10:]                                   # [1, 8160, 1152]
fmap = spatial.reshape(1, 68, 120, -1).permute(0,3,1,2).contiguous()   # [1,1152,68,120] bf16
roi  = roi_align(fmap.float(), boxes,                    # boxes: list[Tensor[100,4]] fp32 xyxy
                 output_size=14,                         # config.json:32 mm_roi_output_size
                 spatial_scale=0.0625,                   # 1/patch_size, _infer_aux_spatial_scale :516-527
                 aligned=True)                           # config.json:33 mm_roi_algined
                                                         # 注意：不传 sampling_ratio => torchvision 默认 -1
# -> [100, 1152, 14, 14] fp32

# object_modules.py:975-980
cat  = torch.cat([roi_l06, roi_l13, roi_l19, roi_l26], dim=1)   # [100, 4608, 14, 14] fp32
out  = cat.mean(dim=(2,3)).reshape(1, 100, 4608)                # [1, 100, 4608] fp32 contiguous
```

boxes 来自 `_scale_roi_boxes`（`object_modules.py:883-899`）：`.to(device, float32).reshape(-1,4).clone()`，x 列乘 `prepared_w/original_w`，y 列乘 `prepared_h/original_h`。参考输入下该比例恰为 1.0（1088/1920 是 16 的倍数），但**不要删掉这一步**：1920×1080 输入时它是真实缩放。传入 roi_align 的是 **list[Tensor[N,4]]（每图一个）**，不是带 batch-index 的 `[N,5]`。

**4608 的通道顺序（level-major，弄反不会报错、不会 NaN，只会静默掉点）：**

| 通道区间 | 内容 |
|---|---|
| `[0:1152]` | blocks[6] |
| `[1152:2304]` | blocks[13] |
| `[2304:3456]` | blocks[19] |
| `[3456:4608]` | blocks[26] |

再往下（TRT 无关，仅供对齐理解）：`multilevel_roi_visual_prompt.py:825` `torch.cat([aux, vt], dim=-1)` ⇒ `[1,100,6656]`，aux 占 `[0:4608]`，VT SimpleFPN 4×512 占 `[4608:6656]`（p2/p3/p4/p5，spatial_scale `[0.25,0.125,0.0625,0.03125]`）。随后**整条 6656 向量加一个正弦位置编码**（`:936-939`，`mm_add_pos_embed=true` + `bbox_based`），再 `.squeeze(0)` 喂 `mm_projector_aux`（`Linear(6656,2048)→GELU→Linear(2048,2048)`，`mm_projector_aux.0.weight` 形状 `[2048,6656]`，列布局被这张权重矩阵硬编码）。
⚠ **验证 TRT 时不要拿最终 6656 的前 4608 列去比**——那里已经叠加了位置编码。要比就比 `aux_out["roi_features"]`（`[1,100,4608]` fp32）或四个 tap 本身。

### 2.7 塔的返回契约（wrapper 必须原样产出）

streamed（部署路径，`object_modules.py:1029-1034`）：

| key | 类型 / 形状 | 参考值 |
|---|---|---|
| `roi_features` | `torch.float32 [1, sum(K_i), 4608]` contiguous CUDA | `[1,100,4608]` |
| `roi_feature_counts` | `list[int]`，长度 = 图 batch | `[100]` |
| `input_size` | `tuple[int,int]` (H,W)，**resize 之后的尺寸** | `(1088, 1920)` |
| `feature_map_size` | `tuple[int,int]` (H,W) | `(68, 120)` |

`roi_feature_counts` 是硬契约：`omchat_qwen3_5_vl.py:999` 校验 `sum(counts) != roi_features.shape[1]` 抛错，`:1012` 按它在 **dim=1** 上切分。
dense（`images` 为 list 时的回退，`_extract_feature_map` `:813-823`）：`{"image_features": [4 × Tensor], "input_size": (H,W)}`，**无** `roi_features`/`feature_map_size`。

---

## 3. 包装类必须实现的接口

### 3.1 推荐实现方式：**继承 `OmChatVisionTowerAux`，只覆盖 block 循环**

```python
class TrtVisionTowerAux(OmChatVisionTowerAux):
    def __init__(self, radio_cfg): super().__init__(radio_cfg); self._engine = None
```

这一决定同时解决了权重加载、`.to()`、conditioner 配置、`config` 属性、varlen 属性、宿主预处理复用五个问题。**不要从 `nn.Module` 从零写。**

### 3.2 外部访问点全表（全部来自继承，除非标「需覆盖」）

| 访问点 | 调用方 | 说明 |
|---|---|---|
| `build_vision_tower_aux(config, model_dir) -> nn.Module` | `omchat_qwen3_5_vl.py:813`（**唯一**构造点） | 唯一需要改的工厂函数：把 `return OmChatVisionTowerAux(radio_cfg)`（`object_modules.py:1532`）换成返回 `TrtVisionTowerAux`。工厂已在 `:1531` 把 `radio_cfg.vitdet_window_size = config.mm_vitdet_window_size`（=None） |
| 属性名必须是 `vision_tower_aux` | `omchat_qwen3_5_vl.py:788,813`；WeightsMapper `:776` `"model.vision_tower_aux." -> "vision_tower_aux."` | AutoWeightsLoader 按属性名路由 |
| `.to(device=, dtype=)` | `:841-843` | 继承。引擎与之无关，但 torch 子模块会被转成 bf16 |
| `.config.hidden_size` | `object_modules.py:579-583`（`build_object_vp_extractor`）⇒ `aux_feature_dims=[1152]` | 继承（`:805-807` property，`:765` 设置 `hidden_size = backbone.embed_dim`） |
| `.config.patch_size` | `_infer_aux_spatial_scale` `:516-527` ⇒ `spatial_scale = 1/16` | 继承。注意该函数先查 `getattr(aux,"spatial_scale",None)`——**wrapper 绝不能新增名为 `spatial_scale` 的属性** |
| `._configure_input_conditioner()` | `:874`（`.to(bf16)` 之后重新调用） | 继承。若你重建/重转模块，**必须**在转 dtype 后再调一次，否则 conditioner 输出 fp32 |
| `.varlen_batch_enabled` | `omchat_qwen3_5_vl.py:1980` | 继承（property `:809`）。若引擎不支持 varlen，**覆盖为返回 False**，可让上游多图请求走逐图分组而非 packed |
| `forward(images, *, roi_boxes=None, roi_output_size=14, roi_spatial_scale=1/16, roi_aligned=True)` | `:1405`（无 box）/ `:1406-1412`（有 box） | **需覆盖分发或其下层** |
| `replace_torch_linear_with_flashrt_nvfp4(self.vision_tower_aux)` | `:859-862`，仅当 `quant_config.get_name()=="flashrt_nvfp4" and scope=="all_linear"` | 会递归替换塔内所有 `nn.Linear`。TRT 版本下必须确保该分支不触发，或在 wrapper 里屏蔽 |
| `get_aux_vision_fp8_targets(config)` / `replace_aux_vision_linear_with_fp8(...)` | `:875-880` | 当前 `aux_vision_quantization_config` 键在两个 config.json 中都不存在 ⇒ 返回空集 ⇒ 不替换 |
| `process_aux_vision_fp8_modules(tower)` / `process_flashrt_nvfp4_torch_linears(tower)` | `:2524-2525`（load_weights 结束后） | 遍历 modules，对 TRT wrapper 是 no-op。**这是构建/refit 引擎的天然挂钩点**（权重此时已全部就位） |

**最小覆盖面（推荐只覆盖这一个方法）**：

```python
def _forward_token_rois_eager(self, tokens, *, grid_size, boxes,
                              output_size, spatial_scale, aligned) -> torch.Tensor:
    f6, f13, f19, f26 = self._run_engine(tokens)          # 4 × [B, L, 1152] bf16
    rois = [self._roi_align_tokens(f, grid_size=grid_size,
                                   prefix_tokens=self._prefix_tokens, boxes=boxes,
                                   output_size=output_size, spatial_scale=spatial_scale,
                                   aligned=aligned)
            for f in (f6, f13, f19, f26)]
    cat = torch.cat(rois, dim=1)
    return cat.mean(dim=(2, 3)).reshape(1, cat.shape[0], cat.shape[1])
```
这样 `_forward_rois_tensor`(:982)、`_prepare_image`、`_prepare_token_inputs`、`_scale_roi_boxes`、`_roi_align_tokens`、返回 dict 全部原封不动，**streamed 契约自动满足**。同时把 `_forward_token_intermediates_eager`(:851) 也覆盖（走同一引擎，返回值需 `[:, prefix:]` 剥前缀），dense/varlen 路径就一并覆盖。

### 3.3 权重加载：330 个 checkpoint 张量怎么办

**问题**：`AutoWeightsLoader._load_module`（`vllm_/model_executor/models/utils.py:293-339`）对任何无法落位的名字抛
`ValueError: There is no module or parameter named {prefix!r}`。checkpoint 里 `model.vision_tower_aux.*` 有 **330** 个张量。

**三种机制，按推荐度排序：**

| # | 机制 | 做法 | 评价 |
|---|---|---|---|
| **1（推荐）** | **保留真实子模块** | `TrtVisionTowerAux` 继承 `OmChatVisionTowerAux`，`self.image_tower = RADIOModel(radio_cfg)` 照旧存在 ⇒ 名字完全对得上，加载零改动 | 零风险。代价：torch 权重与引擎权重并存 862.5 MB × 2 |
| 2 | 自定义 `load_weights` | 给 wrapper 加 `def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]`。`utils.py:272-284` 对非根模块检测 `callable(getattr(module,'load_weights',None))` 并委派；因为 `weights` 是生成器，在方法内消费完后后续 `_groupby_prefix` 循环为空，不会抛错。必须返回已消费的**相对名**集合（调用方用 `_get_qualname(base_prefix, x)` 加前缀） | 省内存，但你得自己把 330 个张量喂给引擎 build/refit，且失去 PyTorch 参考实现 |
| 3 | `skip_prefixes` / `ignore_unexpected` | 改 `omchat_qwen3_5_vl.py:2518` 的 `AutoWeightsLoader(self, skip_prefixes=[...])` | 最脏，且会让权重静默丢失 |

**推荐组合 = 机制 1 + 显存回收**：在 `process_flashrt_nvfp4_torch_linears(self.vision_tower_aux)`（`:2525`，load 完成后）之后构建/refit 引擎，然后释放 block 权重：

```python
def release_torch_blocks(self):
    b = self.image_tower.radio_model.model
    b.blocks = torch.nn.ModuleList()      # 释放 411,466,608 参数 = 822.9 MB
    self._backbone_blocks = ()            # object_modules.py:749 持有强引用，必须一并清空
    self._single_block_parts = ()         # :750-763 同上
```
保留 `patch_generator` + `input_conditioner`（19.77 M 参数 = 39.5 MB），它们仍被 `_prepare_token_inputs` 使用。

**注意**：`default_loader.py:380` 的「权重未初始化」严格检查覆盖不到 aux 塔——`weights_to_load` 在 `model.load_weights()` 之前快照 `named_parameters()`，而 aux 塔是在 `load_weights` 内部由 `_ensure_object_modules()`（`omchat_qwen3_5_vl.py:2510`）懒构造的。所以该检查不会替你兜底。

### 3.4 batch / list 输入的分发（必须实现）

```
forward(images, roi_boxes=...)                     object_modules.py:1493-1515
├─ roi_boxes is not None and not isinstance(images, list)
│     -> _forward_rois_tensor        # B 可能 > 1！引擎 B=1 时 wrapper 必须逐图循环
├─ isinstance(images, list) and varlen_batch_enabled and len>1
│     -> _forward_varlen             # roi_boxes 被静默丢弃，返回 dense dict 列表
├─ isinstance(images, list) 其它
│     -> [self._forward_single(img) for img in images]   # dense dict 列表
└─ 否则 -> _forward_single(images)                       # 单个 dense dict
```
最省事的做法：覆盖 `varlen_batch_enabled -> False`，让 list 输入落到 `[self._forward_single(img) for img in images]`；再让 `_run_engine` 在 `B>1` 时按 batch 维切开逐次 enqueue。这样引擎永远只见 B=1。

---

## 4. 导出可行性与路线

### 4.1 切点 A 下，哪些阻塞被消掉了

| 潜在阻塞 | 切点 A 下状态 |
|---|---|
| CPE 位置编码（`interpolate→crop`，(H,W) 的函数） | **在引擎外**（`patch_generator`） |
| `einops.rearrange`（Im2Patches） | 在引擎外 |
| `F.interpolate(..., antialias=True)`（`_resize_to_supported`） | 在引擎外，且参考输入下恒等 |
| InputConditioner（bf16 舍入常量） | 在引擎外 |
| spectral reparam | **不存在**（`args.spectral_reparam=false`，checkpoint 无相关权重） |
| feature/inter-feature normalizer | **不存在**（两个 config 均为 null） |
| 数据相关控制流 | 无：27 层固定循环，无 early-exit，无 `.item()`（varlen 路径才有，切点 A 不含） |
| forward hook | 无 hook；`forward_intermediates` 由 `MethodType` 绑定（`enable_cpe_support.py:86`），但切点 A 不经过它 |

**剩下的唯一真阻塞：`AuxFlashAttention`。**

### 4.2 AuxFlashAttention 必须在导出时关掉

`OmChatVisionTowerAux.__init__` 默认就会替换 27 层 attention：
```python
# object_modules.py:719-724
requested_varlen_batch = _env_flag("OMCHAT_AUX_VARLEN_BATCH", True)   # 默认 True
use_flash_attention   = _env_flag("OMCHAT_AUX_FLASH_ATTN", True)      # 默认 True
if use_flash_attention or requested_varlen_batch:
    flash_attention_blocks = _replace_aux_attention_with_flash(backbone)
```
`AuxFlashAttention`（`:68`）内部调用 vLLM 的 flash-attn 算子，**不可 ONNX 导出**。导出脚本必须在构造塔之前设：
```
OMCHAT_AUX_FLASH_ATTN=0 OMCHAT_AUX_VARLEN_BATCH=0
```
这样保留 timm 原生 `Attention`（走 `scaled_dot_product_attention`），可导出为标准 ONNX 图。
（日志实证替换生效：`artifacts/logs/vllm_server.log:39` `Enabled vLLM FlashAttention for 27/27 OmChat aux vision blocks (varlen_batch=True)`。）

### 4.3 推荐路线

**主路线：PyTorch → ONNX → TRT engine**

1. 构造一个只含 blocks 的 `nn.Module`：`forward(tokens) -> (blocks[6]out, blocks[13]out, blocks[19]out, blocks[26]out)`，权重从加载好的塔里取。
2. 以 **float32** 导出（`torch.onnx.export`，`dynamo=False`，opset ≥ 17），`dynamic_axes={'tokens': {1: 'L'}, 'feat_l06': {1: 'L'}, ...}`，输出名严格用 `feat_l06/feat_l13/feat_l19/feat_l26`。
   - 不要尝试导出 bf16 ONNX：bfloat16 在 ONNX 里支持零碎。精度由 builder flag 控制，不由 ONNX 权重 dtype 控制。
3. `trtexec` / Python API 构建：`BuilderFlag.BF16`（可加 `FP16`），显式把 network input/output 的 dtype 设为 `DataType.BF16` 并 `setAllowedFormats({TensorFormat.LINEAR})`，避免 TRT 给输出选向量化格式（CHW16 之类）——wrapper 用 `torch.from_blob` 风格零拷贝视图读到向量化 buffer 会得到通道交错的垃圾图（形状对、无 NaN）。
4. 不需要任何 plugin。roi_align 留在 PyTorch。

**理由**：ONNX 自动捕获 LayerNorm 的 `eps` 和 GELU 的具体变体（erf vs tanh），而这两项在本仓库中未被固定（§2.5），手搭网络必须靠猜或靠探针。

**备选路线：TensorRT Network Definition API 手搭**
27 层结构完全一致（`x + attn(LN(x)); x + mlp(LN(x))`，无 LayerScale/DropPath/QK-norm），约 150 行可写完，对 bf16 I/O、格式约束、kernel 选择的控制最强。仅在 ONNX 路线出现 attention 融合不理想或 opset 问题时启用；启用前必须先用探针（§7）确定 `eps` 与 GELU 变体。

**所需 plugin：无。**

TRT 版本要求：需支持 sm_121 的 TensorRT 10.x（GB10 需 10.13 及以上，**未验证**本机安装版本，探针见 §7）。

---

## 5. 与现有加速机制的关系

| 机制 | 位置 | 部署下是否生效 | 对 TRT 设计的含义 |
|---|---|---|---|
| `AuxFlashAttention` | `object_modules.py:68` | **已生效**（`OMCHAT_AUX_FLASH_ATTN` 默认 True；日志 `artifacts/logs/vllm_server.log:39` 27/27） | 434 ms **已经是 flash-attention 的数字**，没有「上 flash attention」这块红利。导出时必须关掉（§4.2）。FA 版本在 GB10 上解析为 **FA2**（selector 特判 major==9/10，sm_121 的 major=12 落到默认 FA2；head_size=72）（**未验证**：需运行时打印 `get_flash_attn_version`） |
| varlen batching | `:1089-1128`, `:1175-1229` | `_varlen_batch_enabled=True`，但只在多图异构尺寸请求下触发 | 引擎不支持 packed 变长时，覆盖 `varlen_batch_enabled -> False`。注意现有 varlen 实现对 `max_seqlen` 调 `.item()`，每 block 一次设备同步（27 次/前向） |
| `AuxFp8Linear` | `:229`，替换器 `:474`，目标解析 `:457`，加载后处理 `omchat_qwen3_5_vl.py:2524` | **未生效**，但**代码是完整可用的** | **更正先前审计结论**：aux 量化**不需要从零写 Linear 类**。`get_aux_vision_fp8_targets` 读的是 **config 属性** `aux_vision_quantization_config`（不是环境变量），要求 `quant_method == "fp8"`；该键在 `resources/vlx-seek-1.5-3b-sanitized/config.json` 与 `C-RADIOv4-SO400M/config.json` 中**都不存在**，全仓 grep 只命中读取处本身 ⇒ 返回空集 ⇒ 零替换。覆盖面 = 108/109 个 Linear：27×(attn.qkv, attn.proj, mlp.fc1, mlp.fc2)；**不覆盖** `patch_generator.embedder`（`ViTPatchLinear`，768→1152）。无 adaptor（`adaptor_names: null`）。名字后缀匹配，FlashAttention 替换后仍能命中（`AuxFlashAttention` 以同名 `qkv`/`proj` 暴露子模块） |
| `FlashRTNvFp4TorchLinear` | `:348`，替换入口 `omchat_qwen3_5_vl.py:859-862` | 仅当 `quant_config.get_name()=="flashrt_nvfp4"` 且 `scope=="all_linear"` | 该分支会把 **aux 塔内所有 nn.Linear** 换成 NVFP4 W4A4。现有 NVFP4 部署包用的是 `modelopt_fp4`，其 `quantization_config.ignore` 明确排除 `vision_tower_aux*`（`resources/vlx-seek-1.5-3b-sanitized-native-nvfp4-w4a4-direct/config.json`；manifest `vision_quantized_count: 0`），所以 aux 塔在两个部署包里都是 BF16。TRT 方案下必须确保该替换不与引擎冲突（引擎权重是构建期烘焙的） |
| CUDA-graph runner | `:1259-1340`，dispatcher `:1438-1447` | 未生效（`OMCHAT_AUX_VIT_CUDAGRAPH` 默认 False；日志无 `Captured aux vision CUDA graph`）；且 streamed 路径根本不经过 dispatcher | 无需考虑。TRT 的 `enqueueV3` 自带 graph 友好性 |
| `torch.compile` 路径 | `_get_compiled_forward_intermediates` `:1342` | 未生效（`OMCHAT_COMPILE_AUX_VIT` 默认 False，且它会反过来关掉 streaming） | 与 TRT 互斥，二选一 |
| 异步 aux 流 | `_maybe_start_async_aux_vision_tower` | 未生效（`OMCHAT_PARALLEL_AUX_VIT` 默认 False） | 正交优化：主塔 283 ms 与 aux 塔 434 ms 目前串行。TRT 落地后可与它叠加 |

**先做一个 5 分钟的对照实验**：在 `config.json` 里加
```json
"aux_vision_quantization_config": {"quant_method": "fp8", "targets": ["mlp", "attn"]}
```
就能在**不写任何新代码**的前提下量到 aux 塔 FP8 weight-only 的收益上界，用来给 TRT FP8 阶段定预期。

---

## 6. 实施顺序与风险

前提：性能上限。当前 ~35 TFLOPS / BF16 峰值 ~125 TFLOPS = 28% 利用率；attention 占 ~54% MACs。理想情况下 TRT BF16 能到 50-60% 利用率 ⇒ 434 ms → **约 220-260 ms**（估算，未实测）。注意 434 ms 里含 4 次 roi_align + mean pool（估算 3-4 ms），引擎最多只能认领 ~430 ms。

| # | 步骤 | 交付物 | 数值验证方法 | 天数 |
|---|---|---|---|---|
| 0 | 跑完 §7 全部探针 | 一份确认清单 | — | 0.5 |
| 1 | 固化 PyTorch 参考 | 脚本：加载塔，对参考图跑一遍，落盘 `tokens[1,8170,1152]`、4 个 tap `[1,8170,1152]`、`roi_features[1,100,4608]`（bf16/fp32 原样 `.npy`） | 自比：脚本两次运行 `torch.equal` | 0.5 |
| 2 | ONNX 导出（`OMCHAT_AUX_FLASH_ATTN=0 OMCHAT_AUX_VARLEN_BATCH=0`） | `aux_blocks.onnx`（fp32，L 动态） | onnxruntime CPU fp32 vs PyTorch fp32：4 个 tap **逐层** `cos > 0.9999`、`max_abs` 相对量级 < 1e-4 | 1 |
| 3 | TRT BF16 引擎，profile 冻结 L=8170 | `.plan` + `_run_engine` | vs 步骤 1 的 bf16 tap：**逐层** `cos ≥ 0.999`、`max_abs / ref_abs_max ≤ 2e-2`（bf16 尾数只有 8 bit，别期待更严）。**四层分别比，不要只比拼接结果** | 2 |
| 4 | 接入 `TrtVisionTowerAux`，覆盖 `_forward_token_rois_eager` | 可起服务 | 端到端比 `aux_out["roi_features"]` `[1,100,4608]`：`cos ≥ 0.999`；再**按 1152 一段切成 4 段分别算 cos**，任一段明显低于其它三段 ⇒ 层序或 tap 点错了 | 1 |
| 5 | 权重释放 + 显存核账 | `release_torch_blocks()` | `nvidia-smi` 前后差 ≈ 823 MB | 0.5 |
| 6 | 性能测量 + profile 调优 | 数字 | `aux_vision_tower` 标签的 ms | 1 |
| 7 | 动态 L（三 profile，可选） | 多 profile `.plan` | 在 512×512 / 1024×768 / 2048×2048 各跑一次步骤 3 的对比 | 1 |
| 8 | FP8（ModelOpt PTQ，显式 Q/DQ） | fp8 `.plan` | 逐层 `cos ≥ 0.99`；同时对比 §5 的 `AuxFp8Linear` 纯 PyTorch FP8 基线 | 2-3 |

合计 BF16 到可用 ≈ **5.5-6 天**，FP8 再加 2-3 天。

### 最高返工风险（唯一一个会逼你重建引擎）

**四个输出的层序/tap 点搞错。** 层序错（比如把 binding index 当输出顺序，TRT 不保证它与 ONNX 输出顺序一致）或 tap 点取在 `attn` 之后而非 block 末尾，产生的张量**形状、dtype、数值范围全部正确**，不抛异常、不出 NaN，只是 grounding/detection 质量静默下降。且 `mm_projector_aux.0.weight [2048,6656]` 的列布局把这个顺序硬编码了。

**强制防御措施：**
1. binding 一律按名字取（`feat_l06/feat_l13/feat_l19/feat_l26`），wrapper 里显式 `[f6, f13, f19, f26]` 组 list。
2. 步骤 4 的验证必须**分 4 段算 cos**，绝不能只看整体。
3. 层序错误的判别特征：整体 cos 仍可能有 0.9x（相邻层特征相关），但**分段 cos 会出现一段远低于其它三段**。

### 次高风险

**不能用端到端任务指标当验收门。** 本 checkpoint 的 object head 是**随机初始化**的：`tensor_provenance.json` 中 `model.mm_projector_aux.*`（4 个）与 `model.object_vp_extractor.simple_fpn.*`（42 个）全部 `source_type: fresh_initialization`（合计 46，与 `sanitization_report.json` 的 `fresh_initialization: 46` 一致）；aux 分支本身（`model.vision_tower_aux.*` 330 个）**全部是 `official_radio`**，权重是真训练过的。NVFP4 包的 provenance 文件与之逐字节相同（sha256 `80edca95ec9e3590…`）。
⇒ **绝对任务指标无意义；只有 PyTorch-vs-TRT 的差分指标有效**（差分可以在任意点取，因为两侧共用同一个随机 projector）。

### 另外两个必须关掉的东西

仓库里有两个 opt-in 校验器用 **`torch.equal`（逐位相等）** 比较并在不等时 `raise`，TRT 永远过不了：
- `_validate_streamed_roi_features`（`object_modules.py:1036-1087`），门 `OMCHAT_VALIDATE_STREAM_AUX_ROI_STRICT`
- `_validate_pure_tensor_outputs`（`:1374-1436`），门 `OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT_STRICT`

两者默认关闭。上线前确认环境里没有设置它们。（注意：`OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT=1` 单独设置是 **no-op**——`_forward_intermediates` `:1443-1447` 在 `use_pure_tensor` 为假时提前 return，根本走不到校验调用。要触发它需同时 `OMCHAT_PURE_TENSOR_AUX_VIT=1` 且 `OMCHAT_STREAM_AUX_ROI=0`。）

---

## 7. 待运行时确认的问题

| # | 问题 | 探针 | 期望值 |
|---|---|---|---|
| 1 | LayerNorm eps 与 GELU 变体（若走手搭 TRT 路线则为阻塞项） | `b = model.vision_tower_aux.image_tower.radio_model.model.blocks[0]; print(b.norm1.eps, type(b.mlp.act), getattr(b.mlp.act,'approximate','n/a'))` | `1e-06`, `<class 'torch.nn.modules.activation.GELU'>`, `'none'` |
| 2 | 结构恒等性（LayerScale / DropPath / QK-norm / final norm） | `print(type(b.ls1), type(b.drop_path1), type(b.attn.q_norm), type(m.norm), m.patch_generator.num_cls_tokens, m.patch_generator.num_registers, m.patch_generator.num_skip, len(m.blocks))` | 四个 `Identity`，`3, 7, 10, 27` |
| 3 | conditioner 实际常量（是否 bf16 舍入） | `c = ...radio_model.input_conditioner; print(c.norm_mean.dtype, c.norm_mean.flatten().tolist(), c.norm_std.flatten().tolist(), c.dtype)` | `float32`, `[0.482421875, 0.45703125, 0.408203125]`, `[0.26953125, 0.26171875, 0.275390625]`, `torch.bfloat16` |
| 4 | `pixel_values_aux` 的真实 dtype / 值域 / 尺寸（决定 /255 和 resize 是否真的恒等） | 在 `object_modules.py:998` 前插 `print(image.dtype, tuple(image.shape), float(image.min()), float(image.max()))` | `torch.uint8, (1,3,1088,1920), 0, 255`（若为 1080 则 resize 非恒等） |
| 5 | 部署实际是否 B=1 / 是否出现 list 输入 | 在 `_forward_rois_tensor` 顶部 `print(int(image.shape[0]), len(boxes))`，并在 `forward` 入口 `print(type(images))` | `1 1`，`Tensor`。若出现 `list` 或 B>1，需按 §3.4 加逐图循环 |
| 6 | FlashAttention 版本与 attention 实现（影响 434 ms 基线归因与导出脚本） | `print(type(m.blocks[0].attn))`；以及 grep 日志 `Enabled vLLM FlashAttention for` | `AuxFlashAttention`；日志 `27/27 ... varlen_batch=True` |
| 7 | 本机 TensorRT 版本是否支持 sm_121 | `python3 -c "import tensorrt; print(tensorrt.__version__)"`；`trtexec --version` | ≥ 10.13 |
| 8 | 实测内存带宽（用于校核 §0.3 的 0.55 ms 估算） | `nvbandwidth` 或一个 `torch` 大张量 copy microbenchmark | ~200-273 GB/s |

---

## 附录：已证伪（不要重新推导）

1. **「out_indices 依赖 Python 银行家舍入，half-up 会得到 14」——假。** `round(13.5)` 在 round-half-even 与 round-half-up 下都等于 14（14 就是偶数邻居）。四个值 6.75→7、13.5→14、20.25→20、27.0→27 与舍入模式无关。不要为这个不存在的 tie 写特殊处理。
2. **「四个中间层是 `[B, N, C]`、三个提取点做同样的 `x[:, prefix:].reshape(...)`」——假。** `_reshape_token_intermediates`（`object_modules.py:1131-1149`）**不含任何前缀切片**，前缀在 `_forward_token_intermediates_eager` 的 return 语句里已被剥掉（`:874-880`）。两个 drop-in 点的前缀契约相反。另：「strip+reshape 是纯 view、几乎零成本」也假——`.permute(0,3,1,2).contiguous()` 是 18.8 MB/层的真实转置拷贝，随后 `.float()` 再拷 37.6 MB/层。
3. **「fp32 转换只发生在 roi_align 边界」——假。** 还有三处 fp32 区域：`_prepare_image` 无条件 `.to(torch.float32)`（`:1170`）；InputConditioner 在 fp32 里算 `(x-mean)/std` 再转 bf16（`input_conditioner.py:31-35` + `object_modules.py:797-803`）；CPE 位置编码的 `F.interpolate(pos_embed.float(), ...)`（`vit_patch_generator.py:256,263`）。
4. **「`OMCHAT_VALIDATE_PURE_TENSOR_AUX_VIT=1` 可以白嫖一份逐层 cosine 报告」——假。** 见 §6 末尾：该 flag 不参与分发，单独设置是 no-op；且两个校验器的 raise 门都是 `torch.equal` 逐位相等，TRT 不可能通过。`_validate_streamed_roi_features` 还是先 cat 成 4608 再比，只输出一个标量 `max_abs`、无 cosine，对层序错误几乎没有诊断力。
5. **「aux 塔的量化需要从零写一个 Linear 类」——假。** `AuxFp8Linear`(`:229`) + `replace_aux_vision_linear_with_fp8`(`:474`) + `get_aux_vision_fp8_targets`(`:457`) + 加载后 `process_aux_vision_fp8_modules`(`omchat_qwen3_5_vl.py:2524`) 是完整通路，覆盖 108/109 个 Linear；唯一缺的是 config 里的 `aux_vision_quantization_config` 键。`FlashRTNvFp4TorchLinear`(`:348`) 同样已覆盖 aux 塔（`quant_config` 为 `flashrt_nvfp4` + `scope=all_linear` 时）。
6. **「`input_size` 用错会造成 1088/1080 的 0.7% 框偏移」——在参考输入下不成立。** 1920×1088 时 resize 恒等，`aux_h/current_h == 1.0`；且 streamed 路径下真正影响结果的框缩放发生在塔内部 `_scale_roi_boxes`（`:883-899`），调用方 `:1559-1561` 算出的 `aux_boxes` 只流向 `boxes_for_pos_emb`，而部署配置（`use_vision_tower_object_feature=True` + `concat` + `bbox_based`）下位置编码取的是 `vt_boxes` 分支（`multilevel_roi_visual_prompt.py:898-908`）。
7. **「1080→1088 是 padding」——假。** 是 `F.interpolate(mode='bilinear', align_corners=False, antialias=True)` 的整图垂直拉伸（`object_modules.py:1477-1483`），证据是 `_scale_roi_boxes` 按 `prepared_h/original_h` 缩放框坐标。且 `get_nearest_supported_resolution` 是**就近取整**，某些尺寸会向下取（如 1076→1072），不是「总是向上补齐」。
