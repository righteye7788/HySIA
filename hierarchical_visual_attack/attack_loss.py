"""层次化视觉辅助攻击损失。"""

from typing import Optional, Sequence

import torch
import torch.nn as nn


class HierarchicalVisualAttackLoss(nn.Module):
    """将层-caption 关系矩阵转为图像侧辅助 loss。

    Args:
        layer_weight_strategy: `uniform`、`fixed_depth` 或 `late_only`。
        lambda_hier: 层次化视觉 loss 权重。
    """

    def __init__(
        self,
        layer_weight_strategy: str = "fixed_depth",
        lambda_hier: float = 0.05,
        layer_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if layer_weight_strategy not in {"uniform", "fixed_depth", "late_only"}:
            raise ValueError(f"Unsupported layer_weight_strategy: {layer_weight_strategy}")
        self.layer_weight_strategy = layer_weight_strategy
        self.lambda_hier = float(lambda_hier)
        self.layer_weights = list(layer_weights) if layer_weights is not None else None

    def build_positive_mask(
        self,
        txt2img,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """根据 `txt2img [M]` 构造 `positive_mask [B,M]`。"""

        txt2img_tensor = torch.as_tensor(txt2img, device=device, dtype=torch.long)
        num_texts = int(txt2img_tensor.numel())
        positive_mask = torch.zeros(batch_size, num_texts, device=device)
        valid = (txt2img_tensor >= 0) & (txt2img_tensor < batch_size)
        if valid.any():
            caption_indices = torch.arange(num_texts, device=device)[valid]
            positive_mask[txt2img_tensor[valid], caption_indices] = 1.0
        return positive_mask

    def compute_layer_weights(
        self,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成归一化层权重 `alpha_l`。"""

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if self.layer_weights is not None:
            if len(self.layer_weights) != num_layers:
                raise ValueError("layer_weights length must match the number of layers.")
            weights = torch.tensor(self.layer_weights, device=device, dtype=dtype)
        elif self.layer_weight_strategy == "uniform":
            weights = torch.ones(num_layers, device=device, dtype=dtype)
        elif self.layer_weight_strategy == "late_only":
            weights = torch.zeros(num_layers, device=device, dtype=dtype)
            weights[-1] = 1.0
        else:
            default = torch.tensor([0.2, 0.2, 0.3, 0.3], device=device, dtype=dtype)
            if num_layers == 4:
                weights = default
            else:
                weights = torch.linspace(1.0, 2.0, steps=num_layers, device=device, dtype=dtype)

        weight_sum = weights.sum().clamp_min(1e-12)
        return weights / weight_sum

    def compute_caption_weights(self, positive_mask: torch.Tensor) -> torch.Tensor:
        """同一图像内正 caption 使用均匀权重。"""

        counts = positive_mask.sum(dim=1, keepdim=True)
        return positive_mask / counts.clamp_min(1.0)

    def forward(self, relation_matrix: torch.Tensor, txt2img) -> torch.Tensor:
        """计算未乘 `lambda_hier` 的标量 `L_hier`。"""

        if relation_matrix.dim() != 3:
            raise ValueError("relation_matrix must have shape [B,L,M].")

        batch_size, num_layers, _ = relation_matrix.shape
        device = relation_matrix.device
        dtype = relation_matrix.dtype
        positive_mask = self.build_positive_mask(txt2img, batch_size, device).to(dtype=dtype)
        caption_weights = self.compute_caption_weights(positive_mask)
        valid_images = (positive_mask.sum(dim=1) > 0).to(dtype=dtype)
        valid_count = valid_images.sum().clamp_min(1.0)

        positive_relations = (relation_matrix * caption_weights.unsqueeze(1)).sum(dim=-1)
        layer_weights = self.compute_layer_weights(num_layers, device, dtype)
        per_image_loss = -(positive_relations * layer_weights.view(1, num_layers)).sum(dim=1)
        return (per_image_loss * valid_images).sum() / valid_count

    def weighted_loss(self, relation_matrix: torch.Tensor, txt2img) -> torch.Tensor:
        """返回可直接加到原始攻击目标中的 `lambda_hier * L_hier`。"""

        return self.lambda_hier * self.forward(relation_matrix, txt2img)
