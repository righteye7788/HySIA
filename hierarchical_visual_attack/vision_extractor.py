"""多层视觉 token 提取器。

该模块只负责从视觉编码器中提取指定层的 token 表征，并统一整理为
`[B, N, C]`。是否聚合 token 由后续关系映射或攻击损失决定。
"""

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalVisionExtractor(nn.Module):
    """使用 forward hook 提取多层视觉 token。

    Args:
        model: VLP 模型。CLIP 使用 `model.visual`，ALBEF/TCL 可使用
            `model.visual_encoder`。
        feature_layers: 需要 hook 的视觉层名称。
        token_mode: `cls`、`patch` 或 `cls_patch`。
        sequence_first: 是否将 `[N, B, C]` 转为 `[B, N, C]`。
        cnn_token_grid_size: CNN 4D feature 的固定空间池化网格。为 `None`
            时保留原始空间尺寸。
    """

    def __init__(
        self,
        model: nn.Module,
        feature_layers: Sequence[str],
        token_mode: str = "patch",
        sequence_first: bool = True,
        cnn_token_grid_size: Optional[int] = None,
    ):
        super().__init__()
        if token_mode not in {"cls", "patch", "cls_patch"}:
            raise ValueError(f"Unsupported token_mode: {token_mode}")
        if cnn_token_grid_size is not None and cnn_token_grid_size <= 0:
            raise ValueError("cnn_token_grid_size must be positive or None.")

        self.model = model
        self.vision_encoder = self._resolve_vision_encoder(model)
        self.feature_layers = list(feature_layers)
        self.token_mode = token_mode
        self.sequence_first = sequence_first
        self.cnn_token_grid_size = cnn_token_grid_size
        self._features: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _resolve_vision_encoder(self, model: nn.Module) -> nn.Module:
        """根据常见 VLP 模型结构找到视觉编码器。"""

        if hasattr(model, "visual"):
            return model.visual
        if hasattr(model, "visual_encoder"):
            return model.visual_encoder
        return model

    def _register_hooks(self) -> None:
        """只在指定层注册 hook，避免重复捕获无关中间层。"""

        modules = dict(self.vision_encoder.named_modules())
        missing_layers = [name for name in self.feature_layers if name not in modules]
        if missing_layers:
            raise KeyError(f"Missing visual feature layers: {missing_layers}")

        for layer_name in self.feature_layers:
            handle = modules[layer_name].register_forward_hook(self._build_hook(layer_name))
            self._hook_handles.append(handle)

    def _build_hook(self, layer_name: str):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self._features[layer_name] = output

        return hook

    def clear_features(self) -> None:
        """每次 forward 前清空缓存，避免复用上一次输入的特征。"""

        self._features.clear()

    def remove_hooks(self) -> None:
        """移除 hook，避免长期实验中重复注册导致内存增长。"""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def to_token_format(self, feature: torch.Tensor) -> torch.Tensor:
        """将 CNN / Transformer 输出统一为 `[B, N, C]`。"""

        has_cls_token = True
        if feature.dim() == 4:
            has_cls_token = False
            if self.cnn_token_grid_size is not None:
                feature = F.adaptive_avg_pool2d(
                    feature,
                    (self.cnn_token_grid_size, self.cnn_token_grid_size),
                )
            feature = feature.flatten(2).transpose(1, 2)
        elif feature.dim() == 3 and self.sequence_first:
            feature = feature.transpose(0, 1)
        elif feature.dim() != 3:
            raise ValueError(f"Unsupported visual feature shape: {tuple(feature.shape)}")

        if self.token_mode == "cls":
            if not has_cls_token:
                return feature.mean(dim=1, keepdim=True)
            return feature[:, :1, :]
        if self.token_mode == "patch":
            if has_cls_token and feature.shape[1] > 1:
                return feature[:, 1:, :]
            return feature
        return feature

    def project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """投影到图文对齐空间并做 L2 归一化。

        CLIP ViT 的中间层 token 与最终输出共享 `ln_post` 和 `proj`。
        对于没有这些层的模型，保留原维度后归一化，作为兼容降级路径。
        """

        visual = self.vision_encoder
        if hasattr(visual, "ln_post"):
            tokens = visual.ln_post(tokens)
        elif hasattr(self.model, "vision_proj") and hasattr(visual, "norm"):
            # ALBEF's block hooks fire before the ViT's final LayerNorm.  The
            # pretrained vision projection expects normalized ViT features.
            tokens = visual.norm(tokens)
        if hasattr(visual, "proj") and visual.proj is not None:
            tokens = tokens @ visual.proj
        elif hasattr(self.model, "vision_proj"):
            tokens = self.model.vision_proj(tokens)
        return F.normalize(tokens, dim=-1)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取并返回 `{layer_name: [B,N,D]}` 多层 token。"""

        self.clear_features()
        _ = self.vision_encoder(images)

        features: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for layer_name in self.feature_layers:
            if layer_name not in self._features:
                raise RuntimeError(f"Hook did not capture visual layer: {layer_name}")
            tokens = self.to_token_format(self._features[layer_name])
            features[layer_name] = self.project_tokens(tokens)
        return features
