"""层-token-caption 关系映射。"""

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCaptionRelationMapper(nn.Module):
    """计算多层视觉 token 与 batch 内 caption 的关系矩阵。

    Args:
        token_attention_tau: token attention 温度。
        token_aggregation: `caption_attention` 或 `uniform`。
    """

    def __init__(
        self,
        token_attention_tau: float = 0.1,
        token_aggregation: str = "caption_attention",
    ):
        super().__init__()
        if token_attention_tau <= 0:
            raise ValueError("token_attention_tau must be positive.")
        if token_aggregation not in {"caption_attention", "uniform"}:
            raise ValueError(f"Unsupported token_aggregation: {token_aggregation}")
        self.token_attention_tau = token_attention_tau
        self.token_aggregation = token_aggregation

    def compute_token_scores(
        self,
        visual_tokens: torch.Tensor,
        txt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """计算 `S [B,N,M] = normalize(z)^T normalize(t)`。"""

        visual_tokens = F.normalize(visual_tokens, dim=-1)
        txt_embeds = F.normalize(txt_embeds, dim=-1)
        return torch.einsum("bnd,md->bnm", visual_tokens, txt_embeds)

    def compute_token_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """根据 token-caption 分数计算 token 权重。"""

        if self.token_aggregation == "uniform":
            return torch.full_like(scores, 1.0 / max(scores.shape[1], 1))
        return torch.softmax(scores / self.token_attention_tau, dim=1)

    def aggregate_relations(
        self,
        scores: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """将 token 分数聚合为 `relations [B,M]`。"""

        return (weights * scores).sum(dim=1)

    def forward(
        self,
        multi_layer_tokens: Dict[str, torch.Tensor],
        txt_embeds: torch.Tensor,
        return_token_weights: bool = False,
    ):
        """返回 `relation_matrix [B,L,M]`，可选返回每层 token 权重。"""

        relations = []
        token_weights = OrderedDict()
        for layer_name, visual_tokens in multi_layer_tokens.items():
            scores = self.compute_token_scores(visual_tokens, txt_embeds)
            weights = self.compute_token_weights(scores)
            relations.append(self.aggregate_relations(scores, weights))
            if return_token_weights:
                token_weights[layer_name] = weights

        relation_matrix = torch.stack(relations, dim=1)
        if return_token_weights:
            return relation_matrix, token_weights
        return relation_matrix
