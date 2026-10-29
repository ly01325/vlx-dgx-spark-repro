import math
import os
import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .simple_fpn import SimpleFP

_SINE_DIM_T_CACHE = {}


def _get_sine_dim_t(dim_of_pos_feats: int, device: torch.device) -> torch.Tensor:
    key = (dim_of_pos_feats, device.type, device.index)
    dim_t = _SINE_DIM_T_CACHE.get(key)
    if dim_t is None:
        dim_t = torch.arange(
            dim_of_pos_feats,
            dtype=torch.float32,
            device=device,
        )
        dim_t = 10000 ** (2 * (dim_t // 2) / dim_of_pos_feats)
        _SINE_DIM_T_CACHE[key] = dim_t
    return dim_t


def _to_dtype_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def _float32_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    return _to_dtype_if_needed(tensor, torch.float32)


def _normalize_boxes_for_position(
    boxes: torch.Tensor,
    original_img_width: float,
    original_img_height: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    boxes = _to_dtype_if_needed(boxes, dtype)
    x1 = boxes[:, 0] / original_img_width
    y1 = boxes[:, 1] / original_img_height
    x2 = boxes[:, 2] / original_img_width
    y2 = boxes[:, 3] / original_img_height
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    return torch.stack((center_x, center_y, width, height), dim=-1)


def _flatten_boxes_for_position(
    boxes: Union[torch.Tensor, List[torch.Tensor]],
) -> torch.Tensor:
    if isinstance(boxes, torch.Tensor):
        if boxes.ndim == 1:
            return boxes.unsqueeze(0)
        return boxes.reshape(-1, boxes.shape[-1])

    if not boxes:
        raise ValueError("boxes for position embedding must not be empty")

    flattened = []
    for item in boxes:
        if item.ndim == 1:
            item = item.unsqueeze(0)
        flattened.append(item.reshape(-1, item.shape[-1]))
    return torch.cat(flattened, dim=0)


def generate_2d_position_embedding(height, width, dim, device):
    """为特征图生成2D位置编码
    
    Args:
        height, width: 特征图的高度和宽度
        dim: 位置编码的维度（应该与特征通道数匹配）
        device: 设备
    
    Returns:
        pos_embed: [H, W, dim] 的位置编码
    """
    # 生成网格坐标
    y_pos = torch.arange(height, dtype=torch.float32, device=device)
    x_pos = torch.arange(width, dtype=torch.float32, device=device)
    
    # 归一化到[0, 1]
    y_pos = y_pos / height
    x_pos = x_pos / width
    
    # 创建网格
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
    
    # 生成位置编码
    scale = 2 * math.pi
    # dim_t = torch.arange(dim // 2, dtype=torch.float32, device=device)
    # dim_t = 10000 ** (2 * (dim_t // 2) / (dim // 2))
    quarter_dim = dim // 4
    dim_t = torch.arange(quarter_dim, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / quarter_dim) if quarter_dim > 0 else torch.tensor([1.0], device=device)
    
    # X方向的编码
    x_embed = x_grid.unsqueeze(-1) * scale
    pos_x = x_embed / dim_t
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
    
    # Y方向的编码
    y_embed = y_grid.unsqueeze(-1) * scale  
    pos_y = y_embed / dim_t
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
    
    # 拼接X和Y方向的编码
    pos_embed = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, dim]
    
    return pos_embed

def gen_sineembed_for_position(pos_tensor, dim_of_pos_feats):
    """Generate sine position embedding from a position tensor.

    Args:
        pos_tensor (torch.Tensor): shape: [batch_size, N, 4]. the last dimension is [cx, cy, w, h] in
            normalized coordinates in range [0, 1].
        out_dim (int): the output dimension of the position embedding.

    Returns:
        pos (torch.Tensor): shape: [batch_size, N, out_dim].
    """
    scale = 2 * math.pi
    dim_t = _get_sine_dim_t(dim_of_pos_feats, pos_tensor.device)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_fourier_embed_for_position(
    pos_tensor: torch.Tensor,
    dim_of_pos_feats: int,
    fourier_B: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
    reorder_to_sine_coord_order: bool = True,
) -> torch.Tensor:
    """Random Fourier feature (RFF) position embedding for continuous box coordinates.

    Maps normalized coordinates :math:`\\mathbf{v} \\in [0, 1]^d` (with :math:`d=2` or :math:`d=4`) to

    .. math::
        \\gamma(\\mathbf{v}) = [\\cos(2\\pi \\mathbf{B}\\mathbf{v}), \\; \\sin(2\\pi \\mathbf{B}\\mathbf{v})],

    where rows of :math:`\\mathbf{B}` are i.i.d. :math:`\\mathcal{N}(0, \\sigma^2)` (Tancik et al., Fourier
    Features Let Networks Learn High Frequency Functions in Low Dimensional Domains). Output last dim
    matches ``gen_sineembed_for_position`` for the same ``pos_tensor`` and ``dim_of_pos_feats``:
    ``d * dim_of_pos_feats``.

    When ``reorder_to_sine_coord_order`` is True (default), :math:`\\mathbf{v}` uses the same axis order
    as the sine helper (``cy, cx`` then ``w, h``), i.e. ``[..., [1, 0]]`` or ``[..., [1, 0, 2, 3]]`` for
    inputs stored as ``[..., cx, cy, w, h]``.

    Args:
        pos_tensor: ``[batch_size, N, d]`` with ``d in {2, 4}``, values in ``[0, 1]``.
        dim_of_pos_feats: Same meaning as in sine embedding; total output dim is ``d * dim_of_pos_feats``.
        fourier_B: Optional fixed matrix of shape ``[M, d]`` with ``M = d * dim_of_pos_feats // 2``.
            If ``None``, :math:`\\mathbf{B}` is sampled on every call (not recommended for training;
            register a buffer in your module and pass it for stable features).
        sigma: Standard deviation of Gaussian entries when ``fourier_B`` is ``None``.
        reorder_to_sine_coord_order: If True, build :math:`\\mathbf{v}` as ``(cy, cx, ...)`` to align with
            ``gen_sineembed_for_position``; if False, use tensor order ``(cx, cy, w, h)``.

    Returns:
        Tensor of shape ``[batch_size, N, d * dim_of_pos_feats]``.
    """
    d_in = pos_tensor.size(-1)
    if d_in not in (2, 4):
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(d_in))

    out_dim = d_in * dim_of_pos_feats
    num_freqs = out_dim // 2
    if num_freqs * 2 != out_dim:
        raise ValueError(
            "d * dim_of_pos_feats must be even for Fourier embedding; got d={}, dim_of_pos_feats={}".format(
                d_in, dim_of_pos_feats
            )
        )

    if reorder_to_sine_coord_order:
        if d_in == 2:
            v = pos_tensor[..., [1, 0]]
        else:
            v = pos_tensor[..., [1, 0, 2, 3]]
    else:
        v = pos_tensor

    if fourier_B is None:
        B = torch.randn(
            num_freqs,
            d_in,
            dtype=pos_tensor.dtype,
            device=pos_tensor.device,
        )
        B = B * sigma
    else:
        B = fourier_B
        if B.shape != (num_freqs, d_in):
            raise ValueError(
                "fourier_B must have shape [{}, {}], got {}".format(num_freqs, d_in, tuple(B.shape))
            )
        if B.device != pos_tensor.device or B.dtype != pos_tensor.dtype:
            B = B.to(device=pos_tensor.device, dtype=pos_tensor.dtype)

    # angles: [B, N, M] = 2π * (v @ B.T)
    angles = (2.0 * math.pi) * torch.matmul(v, B.t())
    pos = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return pos

class FourierPositionalEncoding(nn.Module):
    def __init__(self, d_in: int = 4, dim_of_pos_feats: int = 128, sigma: float = 10.0):
        super().__init__()
        self.dim_of_pos_feats = dim_of_pos_feats
        
        # 提前算好 B 矩阵的形状并固定下来
        out_dim = d_in * dim_of_pos_feats
        num_freqs = out_dim // 2
        
        # 注册为 buffer，它会保存在 state_dict 中并随模型移动 (to_device)，但不会计算梯度
        B = torch.randn(num_freqs, d_in) * sigma
        self.register_buffer("fourier_B", B)

    def forward(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        # 调用你写好的优美函数，并将固定的 buffer 传进去
        return gen_fourier_embed_for_position(
            pos_tensor=pos_tensor,
            dim_of_pos_feats=self.dim_of_pos_feats,
            fourier_B=self.fourier_B, 
            reorder_to_sine_coord_order=True
        )

def gen_sineembed_for_position_v2(pos_tensor, dim_of_pos_feats):
    """Generate sine position embedding from a position tensor.
    Modified to include log-scale transformation for width and height (inspired by Qwen3-VL).

    Args:
        pos_tensor (torch.Tensor): shape: [batch_size, N, 4]. the last dimension is [cx, cy, w, h] in
            normalized coordinates in range [0, 1].
        dim_of_pos_feats (int): the output dimension of the position embedding.

    Returns:
        pos (torch.Tensor): shape: [batch_size, N, out_dim].
    """
    scale = 2 * math.pi
    dim_t = _get_sine_dim_t(dim_of_pos_feats, pos_tensor.device)
    
    # 1. 保持对中心点 cx, cy 的线性映射 (保留原版视觉重心的优势)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
        
    elif pos_tensor.size(-1) == 4:
        # ---------------- 核心修改点开始 ----------------
        
        # 2. 对宽度 w 进行对数尺度变换
        w = pos_tensor[:, :, 2]
        # 注意：这里加了一个极小值 1e-6，是为了防止当 w 恰好为 0 时，log(0) 产生 NaN 导致训练崩溃
        w_log = 0.2 * torch.log(w + 1e-6) + 0.5
        w_embed = w_log * scale
        
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # 3. 对高度 h 进行对数尺度变换
        h = pos_tensor[:, :, 3]
        h_log = 0.2 * torch.log(h + 1e-6) + 0.5
        h_embed = h_log * scale
        
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        
        # ---------------- 核心修改点结束 ----------------

        # 依然保持 Y, X, W, H 的拼接顺序，确保和你后续的其他网络结构（如 MLP）完全兼容
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        
    return pos

class MultiLevelROIVisualPrompt(nn.Module):
    """Initialize the MultiLevelROIVisualPrompt.

    Args:
        output_size (Optional[int]): The size of the output. Default is None.
        channel_per_level (List[int]): List of channels per level. Default is [192, 384, 768, 1536].
        spatial_scale (Optional[float]): The spatial scale factor. Default is None.
        with_additional_projection (bool): Whether to use additional projection. Default is False.
        visual_prompt_hidden_size (int): The hidden size of the visual prompt. Default is 1024.
        add_pos_embedding (bool): Whether to add position embedding. Default is False.
        pos_embedding_dim (int): The dimension of the position embedding. Default is 1024.
    """

    def __init__(
        self,
        output_size: int = None,
        aux_feature_dims: List[int] = None,
        spatial_scale: float = None,
        add_pos_embedding: bool = False,
        pos_embedding_dim: int = 1024,
        use_vision_tower_object_feature: bool = False,
        vision_tower_object_feature_dim: int = 5120,
        vision_tower_spatial_scale: float = 1/14,
        object_feature_combination: str = 'mean',  # 'mean', 'mean_sep_pos', 'mean_aux_pos', 'mean_sep_no_vt_pos', 'concat', 'gated_concat'
        use_vt_object_feature_only: bool = False,
        use_panet: bool = False,
        use_hybrid_encoder: bool = False,
        use_simpleFPN_for_vt: bool = False,
        use_simpleFPN_multiLayer_for_vt: bool = False,
        use_spatial_attention: bool = False, #True,
        use_separate_mlp_for_object: bool = False,
        obj_pooling_type: str = 'mean',  # 'mean', 'max', 'adaptive', 'attention_max'
        pos_embedding_strategy: str = 'bbox_based',  # 'bbox_based', 'feature_map_based', 'hybrid'
        use_multi_scale_roi_align: bool = False,
        apply_object_layer_norm: bool = False,
        roi_algined: bool = False,
        use_simpleFPN_multiLayer_for_vt_aux: bool = False,
        use_simpleFPN_for_vt_aux: bool = False,
        use_pos_embed_v2: bool = False,
        use_fourier_pos_embed: bool = False,
        simpleFPN_out_channels_for_vt: int = 512
    ):
        super(MultiLevelROIVisualPrompt, self).__init__()
        self.output_size = output_size
        self.aux_feature_dims = aux_feature_dims
        self.spatial_scale = spatial_scale
        self.add_pos_embedding = add_pos_embedding
        self.pos_embedding_dim = pos_embedding_dim
        self.use_vision_tower_object_feature = use_vision_tower_object_feature
        self.vision_tower_object_feature_dim = vision_tower_object_feature_dim
        self.vision_tower_spatial_scale = vision_tower_spatial_scale
        self.object_feature_combination = object_feature_combination
        self.use_vt_object_feature_only = use_vt_object_feature_only
        self.use_panet = use_panet
        self.use_hybrid_encoder = use_hybrid_encoder
        self.use_simpleFPN_for_vt = use_simpleFPN_for_vt
        self.use_simpleFPN_multiLayer_for_vt = use_simpleFPN_multiLayer_for_vt
        self.use_spatial_attention = use_spatial_attention
        self.use_separate_mlp_for_object = use_separate_mlp_for_object
        self.obj_pooling_type = obj_pooling_type
        self.pos_embedding_strategy = pos_embedding_strategy
        self.use_multi_scale_roi_align = use_multi_scale_roi_align
        self.apply_object_layer_norm = apply_object_layer_norm
        self.roi_algined = roi_algined
        self.vt_roi_sampling_ratio = int(
            os.getenv("OMCHAT_VT_ROI_SAMPLING_RATIO", "-1")
        )
        self.use_simpleFPN_multiLayer_for_vt_aux = use_simpleFPN_multiLayer_for_vt_aux
        self.aux_feature_dims = aux_feature_dims
        self.use_simpleFPN_for_vt_aux = use_simpleFPN_for_vt_aux
        self.use_pos_embed_v2 = use_pos_embed_v2
        self.use_fourier_pos_embed = use_fourier_pos_embed
        self._compile_simple_fpn = os.getenv(
            "OMCHAT_COMPILE_SIMPLE_FPN", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self._compile_simple_fpn_dynamic = os.getenv(
            "OMCHAT_COMPILE_SIMPLE_FPN_DYNAMIC", "1"
        ).lower() in {"1", "true", "yes", "on"}
        self._fused_simple_fpn = os.getenv(
            "OMCHAT_FUSED_SIMPLE_FPN", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self._compiled_simple_fpn_forward = None
        self._simple_fpn_compile_failed = False

        print(f"output_size: {self.output_size} use_vision_tower_object_feature: {self.use_vision_tower_object_feature} vision_tower_object_feature_dim: {self.vision_tower_object_feature_dim} \
              add_pos_embedding: {self.add_pos_embedding} object_feature_combination: {self.object_feature_combination} pos_embedding_dim: {self.pos_embedding_dim} use_vt_object_feature_only: {self.use_vt_object_feature_only} \
              use_panet: {self.use_panet} use_hybrid_encoder: {self.use_hybrid_encoder} use_simpleFPN_for_vt: {self.use_simpleFPN_for_vt} simple_fpn_out_channels_for_vt: {simpleFPN_out_channels_for_vt} use_spatial_attention: {self.use_spatial_attention} obj_pooling_type: {self.obj_pooling_type} pos_embedding_strategy: {self.pos_embedding_strategy} \
              use_multi_scale_roi_align: {self.use_multi_scale_roi_align} apply_object_layer_norm: {self.apply_object_layer_norm} roi_algined: {self.roi_algined} use_simpleFPN_multiLayer_for_vt: {self.use_simpleFPN_multiLayer_for_vt} \
              use_simpleFPN_multiLayer_for_vt_aux: {self.use_simpleFPN_multiLayer_for_vt_aux} use_simpleFPN_for_vt_aux: {self.use_simpleFPN_for_vt_aux} aux_feature_dims: {self.aux_feature_dims} use_pos_embed_v2: {self.use_pos_embed_v2} use_fourier_pos_embed: {self.use_fourier_pos_embed}")
        
        if self.use_fourier_pos_embed:
            self.fourier_pos_embed = FourierPositionalEncoding(d_in=4, dim_of_pos_feats=self.pos_embedding_dim//4, sigma=10.0)
        
        if self.use_vision_tower_object_feature and self.object_feature_combination in ['mean', 'mean_sep_pos', 'mean_aux_pos', 'mean_sep_no_vt_pos']:
            self.vision_tower_object_feature_projector = nn.Sequential(
                nn.Linear(vision_tower_object_feature_dim, pos_embedding_dim),
                nn.GELU(),
                nn.Linear(pos_embedding_dim, pos_embedding_dim)
            )
        
        if self.use_vision_tower_object_feature and self.object_feature_combination in ['gated_concat']:
            self.fusion_gate = GatedFeatureFusion(feature_dim=pos_embedding_dim)
        
        if self.use_panet:
            from .PANet import PANet

            self.panet = PANet(
                self.aux_feature_dims,
                fpn_channels=512,
                use_bn=True,
                use_relu=True,
                dropout=0.0,
            )

        elif self.use_hybrid_encoder:
            from .hybrid_encoder import HybridEncoder

            self.hybrid_encoder = HybridEncoder(hidden_dim=512)
            
        
        if self.use_simpleFPN_for_vt:
            # self.simple_fpn_dim = 768
            # self.simple_fpn_scale_factors = [4.0, 2.0, 1.0, 0.5]
            # self.simple_fpn_stride = 16

            self.simple_fpn_dim = self.vision_tower_object_feature_dim
            self.simple_fpn_scale_factors = [8.0, 4.0, 2.0, 1.0]
            self.simple_fpn_stride = 32
            self.simple_fpn_out_channels = simpleFPN_out_channels_for_vt

            # self.simple_fpn_dim = 1024      # a3  v143
            # self.simple_fpn_scale_factors = [8.0, 4.0, 2.0, 1.0]
            # self.simple_fpn_stride = 32
            # self.simple_fpn_out_channels = 1024


            self.simple_fpn = SimpleFP(out_channels=self.simple_fpn_out_channels, norm="LN", square_pad=0, dim=self.simple_fpn_dim, stride=self.simple_fpn_stride, scale_factors=self.simple_fpn_scale_factors)
            self.simple_fpn_spatial_scales = [
                scale_factor / self.simple_fpn_stride
                for scale_factor in self.simple_fpn_scale_factors
            ]
        elif self.use_simpleFPN_multiLayer_for_vt:
            from .simple_fpn_multi import MultiLayerSimpleFP

            self.simple_fpn_dim = self.vision_tower_object_feature_dim
            self.simple_fpn_scale_factors = [4.0, 2.0, 1.0, 1.0]
            self.simple_fpn_stride = 32
            self.simple_fpn_out_channels = simpleFPN_out_channels_for_vt
            self.simple_fpn = MultiLayerSimpleFP(out_channels=self.simple_fpn_out_channels, norm="LN", square_pad=0, dim=self.simple_fpn_dim, stride=self.simple_fpn_stride, scale_factors=self.simple_fpn_scale_factors)
            self.simple_fpn_spatial_scales = [
                scale_factor / self.simple_fpn_stride
                for scale_factor in self.simple_fpn_scale_factors
            ]

        if self.use_simpleFPN_for_vt and self._fused_simple_fpn:
            self.simple_fpn.enable_channels_last()
        
        if self.use_simpleFPN_for_vt_aux:
            self.simple_fpn_dim_aux = self.aux_feature_dims[0]   # C-RADIOv4 1152
            self.simple_fpn_scale_factors_aux = [4.0, 2.0, 1.0, 0.5]
            self.simple_fpn_stride_aux = 16
            self.simple_fpn_out_channels_aux = 1024
            self.simple_fpn_aux = SimpleFP(out_channels=self.simple_fpn_out_channels_aux, norm="LN", square_pad=0, dim=self.simple_fpn_dim_aux, stride=self.simple_fpn_stride_aux, scale_factors=self.simple_fpn_scale_factors_aux)
            self.simple_fpn_spatial_scales_aux = [
                scale_factor / self.simple_fpn_stride_aux
                for scale_factor in self.simple_fpn_scale_factors_aux
            ]
            print(f'Use SimpleFP for VT_Aux. simple_fpn_dim_aux:{self.simple_fpn_dim_aux}, simple_fpn_scale_factors_aux:{self.simple_fpn_scale_factors_aux}, simple_fpn_stride_aux:{self.simple_fpn_stride_aux}, simple_fpn_out_channels_aux:{self.simple_fpn_out_channels_aux}')
        elif self.use_simpleFPN_multiLayer_for_vt_aux:
            from .simple_fpn_multi import MultiLayerSimpleFP

            self.simple_fpn_dim_aux = self.aux_feature_dims[0]   # DinoV3-H 1280  C-RADIOv4 1152
            self.simple_fpn_scale_factors_aux = [4.0, 2.0, 1.0, 0.5]
            self.simple_fpn_stride_aux = 16
            self.simple_fpn_out_channels_aux = 1024
            self.simple_fpn_aux = MultiLayerSimpleFP(out_channels=self.simple_fpn_out_channels_aux, norm="LN", square_pad=0, dim=self.simple_fpn_dim_aux, stride=self.simple_fpn_stride_aux, scale_factors=self.simple_fpn_scale_factors_aux)
            self.simple_fpn_spatial_scales_aux = [
                scale_factor / self.simple_fpn_stride_aux
                for scale_factor in self.simple_fpn_scale_factors_aux
            ]
            print(f'Use MultiLayerSimpleFP for VT_Aux. simple_fpn_dim_aux:{self.simple_fpn_dim_aux}, simple_fpn_scale_factors_aux:{self.simple_fpn_scale_factors_aux}, simple_fpn_stride_aux:{self.simple_fpn_stride_aux}, simple_fpn_out_channels_aux:{self.simple_fpn_out_channels_aux}')
        
        if self.use_vision_tower_object_feature and self.use_separate_mlp_for_object:
            self.object_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )
            self.aux_object_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )

    def _apply_feature_map_position_embedding(self, features):
        """在特征图级别应用位置编码"""
        enhanced_features = []
        
        for level_idx, feature in enumerate(features):
            if self.add_pos_embedding and self.pos_embedding_strategy in ['feature_map_based', 'hybrid']:
                B, C, H, W = feature.shape
                
                # 直接生成与特征通道数匹配的位置编码
                pos_embed = generate_2d_position_embedding(
                    H, W, C, feature.device
                )  # [H, W, C]
                
                # 直接转换维度并添加到特征上
                pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                feature = feature + _to_dtype_if_needed(pos_embed, feature.dtype)
            
            enhanced_features.append(feature)
        
        return enhanced_features

    def _apply_pooling(self, level_roi_feat):
        """统一的池化方法"""
        if self.obj_pooling_type == 'max':
            return F.adaptive_max_pool2d(level_roi_feat, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.obj_pooling_type == 'adaptive':
            mean_feat = level_roi_feat.mean(dim=(2, 3))
            max_feat = F.adaptive_max_pool2d(level_roi_feat, (1, 1)).squeeze(-1).squeeze(-1)
            return 0.3 * mean_feat + 0.7 * max_feat  # 偏向max pooling
        else:  # default to mean
            return level_roi_feat.mean(dim=(2, 3))

    def _run_simple_fpn(self, features):
        if not self._compile_simple_fpn or self._simple_fpn_compile_failed:
            return self.simple_fpn(features)
        if self._compiled_simple_fpn_forward is None:
            mode = os.getenv("OMCHAT_COMPILE_SIMPLE_FPN_MODE", "default")
            try:
                self._compiled_simple_fpn_forward = torch.compile(
                    self.simple_fpn.forward,
                    mode=mode,
                    fullgraph=True,
                    dynamic=self._compile_simple_fpn_dynamic,
                )
            except Exception as exc:
                self._simple_fpn_compile_failed = True
                warnings.warn(
                    f"Failed to initialize SimpleFPN torch.compile: {exc}",
                    stacklevel=2,
                )
                return self.simple_fpn(features)
        try:
            return self._compiled_simple_fpn_forward(features)
        except Exception as exc:
            self._simple_fpn_compile_failed = True
            warnings.warn(
                f"SimpleFPN torch.compile failed; using eager execution: {exc}",
                stacklevel=2,
            )
            return self.simple_fpn(features)

    def prepare_vt_fpn_features(self, features):
        if not self.use_simpleFPN_for_vt:
            raise RuntimeError("SimpleFPN is not enabled for main vision features")
        return self._run_simple_fpn(features)

    def extract_vt_object_feature(
        self,
        multi_level_features,
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        prepared_fpn_features=None,
    ) -> torch.Tensor:
        if self.use_simpleFPN_for_vt or self.use_simpleFPN_multiLayer_for_vt:
            if prepared_fpn_features is None:
                multi_level_features = self._run_simple_fpn(multi_level_features)
            else:
                multi_level_features = prepared_fpn_features

            roi_features_per_level = []
            
            # feature_strides = [3.5, 7, 14, 28 ]  # 根据你的网络结构调整
            # feature_strides = [8, 16, 32, 32]
            # feature_strides = [8, 16, 32, 64]
            
            for level_idx, level_feature in enumerate(multi_level_features):
                level_roi_feat = roi_align(
                    _float32_if_needed(level_feature),
                    boxes,
                    output_size=self.output_size,
                    spatial_scale=self.simple_fpn_spatial_scales[level_idx],
                    sampling_ratio=self.vt_roi_sampling_ratio,
                    aligned=self.roi_algined
                )
                
                # 使用空间注意力池化替代简单mean
                if self.use_spatial_attention and hasattr(self, 'vt_spatial_attention'):
                    level_roi_feat = self.vt_spatial_attention[level_idx](level_roi_feat)
                else:
                    level_roi_feat = level_roi_feat.mean(dim=(2, 3))
                
                roi_features_per_level.append(level_roi_feat)
            
            out_box_feat = torch.cat(roi_features_per_level, dim=1).unsqueeze(0)
        else:
            concat_multi_level_feature = torch.cat(multi_level_features, dim=1)

            out_box_feat = roi_align(
                _float32_if_needed(concat_multi_level_feature),
                boxes,
                output_size=self.output_size,
                spatial_scale=self.vision_tower_spatial_scale,
                aligned=self.roi_algined
            )

            # Average Pooling -> n,c -> 1,n,c
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )
            if self.object_feature_combination in ['mean', 'mean_sep_pos', 'mean_aux_pos', 'mean_sep_no_vt_pos']:
                out_box_feat = self.vision_tower_object_feature_projector(
                    _to_dtype_if_needed(
                        out_box_feat,
                        concat_multi_level_feature.dtype,
                    )
                ).float()
        return out_box_feat

    def extract_aux_object_feature(
        self,
        multi_level_features: List[torch.Tensor],
        boxes: Union[torch.Tensor, List[torch.Tensor]],
    ) -> tuple[torch.Tensor, int, int]:
        if self.use_simpleFPN_multiLayer_for_vt_aux or self.use_simpleFPN_for_vt_aux:
            multi_level_features = self.simple_fpn_aux(multi_level_features)
            max_height = max(feature.shape[2] for feature in multi_level_features)
            max_width = max(feature.shape[3] for feature in multi_level_features)
            roi_features_per_level = []
            for level_idx, level_feature in enumerate(multi_level_features):
                level_roi_feat = roi_align(
                    _float32_if_needed(level_feature),
                    boxes,
                    output_size=self.output_size,
                    spatial_scale=self.simple_fpn_spatial_scales_aux[level_idx],
                    aligned=self.roi_algined,
                )
                roi_features_per_level.append(level_roi_feat.mean(dim=(2, 3)))
            out_box_feat = torch.cat(roi_features_per_level, dim=1).unsqueeze(0)
            return out_box_feat, max_height, max_width

        max_height = max(feature.shape[2] for feature in multi_level_features)
        max_width = max(feature.shape[3] for feature in multi_level_features)
        if self.pos_embedding_strategy in ["feature_map_based", "hybrid"]:
            multi_level_features = self._apply_feature_map_position_embedding(
                multi_level_features
            )

        same_spatial_shape = all(
            feature.shape[-2:] == (max_height, max_width)
            for feature in multi_level_features
        )
        if same_spatial_shape:
            out_box_feat = torch.cat(
                [
                    roi_align(
                        _float32_if_needed(feature),
                        boxes,
                        output_size=self.output_size,
                        spatial_scale=self.spatial_scale,
                        aligned=self.roi_algined,
                    )
                    for feature in multi_level_features
                ],
                dim=1,
            )
        else:
            concat_multi_level_feature = torch.cat(
                [
                    F.interpolate(
                        _float32_if_needed(feature),
                        size=(max_height, max_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if feature.shape[-2:] != (max_height, max_width)
                    else _float32_if_needed(feature)
                    for feature in multi_level_features
                ],
                dim=1,
            )
            out_box_feat = roi_align(
                concat_multi_level_feature,
                boxes,
                output_size=self.output_size,
                spatial_scale=self.spatial_scale,
                aligned=self.roi_algined,
            )
        out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
            1,
            out_box_feat.shape[0],
            out_box_feat.shape[1],
        )
        return out_box_feat, max_height, max_width
    
    def forward(
        self,
        multi_level_features: List[torch.Tensor],
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        vt_multi_level_features = None,
        vt_boxes: Union[torch.Tensor, List[torch.Tensor]] = None,
        boxes_for_pos_emb: Union[torch.Tensor, List[torch.Tensor]] = None,
        vt_boxes_for_pos_emb: Union[torch.Tensor, List[torch.Tensor]] = None,
        add_pos_embed: bool = True,
        vt_fpn_features=None,
        aux_roi_features: torch.Tensor | None = None,
        aux_feature_map_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Performs Region of Interest (RoI) Align operator on multi-level features. The RoI
        feature on each scale will go through a different linear layer for projection. Different
        RoI features will be summed up and then average pooled.

        Args:
            multi_level_features (Listp[Tensor[N, C, H, W]]): Feature maps from different levels
            boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
                format where the regions will be taken from.
        Returns:
            Tensor[1, K, C]: The output tensor that has the shape KxC, where K is the number of RoIs
        """
        if boxes_for_pos_emb is None:
            boxes_for_pos_emb = boxes
        if vt_boxes_for_pos_emb is None:
            vt_boxes_for_pos_emb = vt_boxes
            # print('----------------------vt_boxes_for_pos_emb is None')

        if self.use_vt_object_feature_only:
            out_box_feat = self.extract_vt_object_feature(vt_multi_level_features, vt_boxes)

            if self.add_pos_embedding and add_pos_embed:
                vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                original_img_width = vt_max_width / self.vision_tower_spatial_scale
                original_img_height = vt_max_height / self.vision_tower_spatial_scale
                pos_boxes = _normalize_boxes_for_position(
                    _flatten_boxes_for_position(vt_boxes_for_pos_emb),
                    original_img_width,
                    original_img_height,
                    out_box_feat.dtype,
                )
                if self.use_pos_embed_v2:
                    pos_embed = gen_sineembed_for_position_v2(pos_boxes.unsqueeze(0), self.pos_embedding_dim // 4)
                else:
                    pos_embed = gen_sineembed_for_position(pos_boxes.unsqueeze(0), self.pos_embedding_dim // 4)
                out_box_feat = out_box_feat + pos_embed
            return out_box_feat
        
        if isinstance(boxes, torch.Tensor):
            boxes = _float32_if_needed(boxes)
        else:
            boxes = [_float32_if_needed(item) for item in boxes]
        if aux_roi_features is None:
            out_box_feat, max_height, max_width = self.extract_aux_object_feature(
                multi_level_features,
                boxes,
            )
        else:
            out_box_feat = aux_roi_features
            if aux_feature_map_size is None:
                max_height = max_width = 0
            else:
                max_height, max_width = aux_feature_map_size

        if self.apply_object_layer_norm:
            out_box_feat = self.aux_object_norm.float()(out_box_feat)
        
        if self.use_vision_tower_object_feature:
            out_box_vt_feat = self.extract_vt_object_feature(
                vt_multi_level_features,
                vt_boxes,
                prepared_fpn_features=vt_fpn_features,
            )
            if self.apply_object_layer_norm:
                out_box_vt_feat = self.vt_object_norm.float()(out_box_vt_feat)
            if self.object_feature_combination in ['mean', 'mean_aux_pos']:
                out_box_feat = (out_box_feat + out_box_vt_feat) / 2      #todo try different combination
            elif self.object_feature_combination in ['concat', 'concat_aux_pos']:
                if self.use_separate_mlp_for_object:
                    original_vt_dtype = out_box_vt_feat.dtype
                    original_aux_dtype = out_box_feat.dtype
                    out_box_vt_feat = self.object_mlp(
                        _to_dtype_if_needed(
                            out_box_vt_feat,
                            self.object_mlp[0].weight.dtype,
                        )
                    ).to(original_vt_dtype)
                    out_box_feat = self.aux_object_mlp(
                        _to_dtype_if_needed(
                            out_box_feat,
                            self.aux_object_mlp[0].weight.dtype,
                        )
                    ).to(original_aux_dtype)
                out_box_feat = torch.cat([out_box_feat, out_box_vt_feat], dim=-1)
            elif self.object_feature_combination in ['gated_concat']:
                original_vt_dtype = out_box_vt_feat.dtype
                gate_dtype = self.fusion_gate.gate_net[0].weight.dtype
                out_box_feat = self.fusion_gate(
                    _to_dtype_if_needed(out_box_feat, gate_dtype),
                    _to_dtype_if_needed(out_box_vt_feat, gate_dtype),
                ).to(original_vt_dtype)
            elif self.object_feature_combination in ['concat_sep_pos', 'mean_sep_pos', 'concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
                # 为out_box_feat和out_box_vt_feat分别计算位置嵌入
                # 使用aux vt的边界框计算位置嵌入
                vt_dim = 5120 if self.object_feature_combination == 'concat_sep_pos' else 2880

                aux_original_img_width = max_width / self.spatial_scale
                aux_original_img_height = max_height / self.spatial_scale
                aux_pos_boxes = _normalize_boxes_for_position(
                    _flatten_boxes_for_position(boxes_for_pos_emb),
                    aux_original_img_width,
                    aux_original_img_height,
                    out_box_feat.dtype,
                )
                if self.use_pos_embed_v2:
                    aux_pos_embed = gen_sineembed_for_position_v2(
                        aux_pos_boxes.unsqueeze(0), 2880 // 4
                    )
                else:
                    aux_pos_embed = gen_sineembed_for_position(
                    aux_pos_boxes.unsqueeze(0), 2880 // 4
                )
                out_box_feat = out_box_feat + aux_pos_embed
                
                # vt object feature 不添加位置嵌入
                if self.object_feature_combination in ['concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
                    pass
                # vt object feature 添加位置嵌入
                else:
                    # 使用vt的边界框计算位置嵌入
                    vt_max_height = max([feature.shape[2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[3] for feature in vt_multi_level_features])
                    vt_original_img_width = vt_max_width / self.vision_tower_spatial_scale
                    vt_original_img_height = vt_max_height / self.vision_tower_spatial_scale
                    vt_pos_boxes = _normalize_boxes_for_position(
                        _flatten_boxes_for_position(vt_boxes_for_pos_emb),
                        vt_original_img_width,
                        vt_original_img_height,
                        out_box_vt_feat.dtype,
                    )
                    if self.use_pos_embed_v2:
                        vt_pos_embed = gen_sineembed_for_position_v2(
                            vt_pos_boxes.unsqueeze(0), vt_dim // 4
                        )
                    else:
                        vt_pos_embed = gen_sineembed_for_position(
                        vt_pos_boxes.unsqueeze(0), vt_dim // 4
                    )
                    out_box_vt_feat = out_box_vt_feat + vt_pos_embed
                
                # 将两个特征连接起来
                if self.object_feature_combination in ['concat_sep_pos', 'concat_sep_no_vt_pos']:
                    out_box_feat = torch.cat([out_box_feat, out_box_vt_feat], dim=-1)
                elif self.object_feature_combination in ['mean_sep_pos', 'mean_sep_no_vt_pos']:
                    out_box_feat = (out_box_feat + out_box_vt_feat) / 2
                

        if self.add_pos_embedding and add_pos_embed and self.object_feature_combination not in ['concat_sep_pos', 'mean_sep_pos', 'concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
            # 根据位置编码策略决定是否应用bbox级别的位置编码
            apply_bbox_pos_embed = (
                self.pos_embedding_strategy == 'bbox_based' or 
                (self.pos_embedding_strategy == 'hybrid' and not (self.use_panet or self.use_hybrid_encoder))
            )
            
            if apply_bbox_pos_embed:
                # 使用vt的边界框计算位置嵌入，而不是aux vt的
                if self.use_vision_tower_object_feature and vt_boxes is not None and self.object_feature_combination not in ['concat_aux_pos', 'mean_aux_pos']:
                    # 如果使用vt，且vt_boxes可用，则使用vt_boxes计算位置嵌入
                    pos_box_source = _flatten_boxes_for_position(
                        vt_boxes_for_pos_emb
                    )
                    # 假设vt_multi_level_features中的特征图大小是vt的输出
                    vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                    # 使用vt的空间尺度（可能需要通过参数传入或另外计算）
                    # 这里假设有一个vt_spatial_scale参数，如果没有，需要适当调整
                    vt_spatial_scale = self.vision_tower_spatial_scale  # 这里使用一个示例值，实际应根据vt的配置设置
                    
                    original_img_width = vt_max_width / vt_spatial_scale
                    original_img_height = vt_max_height / vt_spatial_scale
                else:
                    # 如果不使用vt或vt_boxes不可用，则继续使用aux vt的boxes
                    max_width = max([feature.shape[3] for feature in multi_level_features])
                    max_height = max([feature.shape[2] for feature in multi_level_features])
                    pos_box_source = _flatten_boxes_for_position(boxes_for_pos_emb)
                    original_img_width = max_width / self.spatial_scale
                    original_img_height = max_height / self.spatial_scale

                pos_boxes = _normalize_boxes_for_position(
                    pos_box_source,
                    original_img_width,
                    original_img_height,
                    out_box_feat.dtype,
                )
                if self.use_pos_embed_v2:
                    pos_embed = gen_sineembed_for_position_v2(
                        pos_boxes.unsqueeze(0), self.pos_embedding_dim // 4
                    )
                elif self.use_fourier_pos_embed:
                    pos_embed = _to_dtype_if_needed(
                        self.fourier_pos_embed(pos_boxes.unsqueeze(0)),
                        out_box_feat.dtype,
                    )
                else:
                    pos_embed = gen_sineembed_for_position(
                    pos_boxes.unsqueeze(0), self.pos_embedding_dim // 4
                )
                out_box_feat = out_box_feat + pos_embed

        return out_box_feat

class GatedFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        total_dim = feature_dim
        
        # Gate 网络输入是拼接后的特征 (total_dim)
        # 输出也是 total_dim，这样才能为每一个通道都分配一个权重
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, total_dim), 
            nn.Sigmoid()
        )
        
        # 初始化 bias，让初始状态接近 concat (gate值接近 0.5 或特定比例)
        # 如果想让模型一开始稍微偏向 primary，可以把 bias 设大一点
        nn.init.constant_(self.gate_net[-2].bias, 0.0) 

    def forward(self, primary_feat, aux_feat):
        # primary_feat: [B, N, primary_dim] (e.g., 3000)
        # aux_feat:     [B, N, aux_dim]     (e.g., 2000)
        
        # 1. 拼接作为上下文输入
        combined = torch.cat([primary_feat, aux_feat], dim=-1) # [B, N, 5000]
        
        # 2. 计算 Gate 权重 (Total Dimension)
        gate_values = self.gate_net(combined) # [B, N, 5000]
        
        # 3. [关键修正] 切分 Gate 权重
        # 把 5000 维的权重切成 3000 (给Primary) 和 2000 (给Aux)
        gate_primary, gate_aux = torch.split(gate_values, [primary_feat.shape[-1], aux_feat.shape[-1]], dim=-1)
        
        # 4. 分别加权
        # Primary 部分使用前半部分权重
        gated_primary = primary_feat * gate_primary 
        
        # Aux 部分通常有两种策略：
        # 策略 A (独立门控): 直接用后半部分权重。这意味着模型可以同时增强或同时抑制两者。
        gated_aux = aux_feat * gate_aux
        
        # 策略 B (互补门控 - 之前的写法，但维度不对需调整): 
        # 如果你想强制 trade-off，通常不适用于维度不同的情况。
        # 建议使用策略 A，让模型自由决定每一维的重要性。
        
        # 5. 拼接输出 (保持总维度 5000，兼容 Projector)
        out = torch.cat([gated_primary, gated_aux], dim=-1)
        
        return out
