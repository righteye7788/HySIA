"""可学习双曲 Adapter。

该模块只负责将冻结 VLP encoder 产出的视觉 token 与文本特征映射到
原点切空间和 Poincare ball。Adapter 可在 clean 图文数据上离线训练；
攻击实施阶段通过 checkpoint 加载并冻结。
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import PoincareBall


@dataclass
class LearnableHyperbolicAdapterConfig:
    """构建 `LearnableHyperbolicAdapter` 所需的最小结构配置。"""

    layer_dims: Dict[str, int]
    text_dim: int
    adapter_dim: int = 256
    hidden_dim: Optional[int] = None
    radius_min: float = 0.1
    radius_max: float = 0.8
    input_normalization: str = "layernorm_l2"
    curvature: float = 1.0
    eps: float = 1e-5


def _safe_module_key(layer_name: str) -> str:
    """将真实层名转成可用于 `ModuleDict` 的稳定 key。"""

    return layer_name.replace(".", "__dot__")


class _ProjectionHead(nn.Module):
    """Linear-LayerNorm-GELU-Linear 轻量映射头。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LearnableHyperbolicAdapter(nn.Module):
    """可学习方向头与半径头组成的双曲映射 Adapter。"""

    def __init__(self, config: LearnableHyperbolicAdapterConfig):
        super().__init__()
        self.config = config
        self.layer_names = list(config.layer_dims.keys())
        self.layer_name_to_key = {name: _safe_module_key(name) for name in self.layer_names}
        self.radius_min = float(config.radius_min)
        self.radius_max = float(config.radius_max)
        self.input_normalization = config.input_normalization
        self.ball = PoincareBall(curvature=config.curvature, eps=config.eps)

        if self.radius_min <= 0 or self.radius_min >= self.radius_max:
            raise ValueError("radius_min and radius_max must satisfy 0 < min < max.")
        if self.input_normalization not in {"none", "l2", "layernorm_l2"}:
            raise ValueError(f"Unsupported adapter input normalization: {self.input_normalization}")

        self.visual_direction_heads = nn.ModuleDict()
        self.visual_radius_heads = nn.ModuleDict()
        hidden_dim = config.hidden_dim or config.adapter_dim
        for layer_name, input_dim in config.layer_dims.items():
            key = self.layer_name_to_key[layer_name]
            self.visual_direction_heads[key] = _ProjectionHead(
                input_dim,
                config.adapter_dim,
                hidden_dim,
            )
            self.visual_radius_heads[key] = _ProjectionHead(
                input_dim,
                1,
                hidden_dim,
            )

        self.text_direction_head = _ProjectionHead(
            config.text_dim,
            config.adapter_dim,
            hidden_dim,
        )
        self.text_radius_head = _ProjectionHead(
            config.text_dim,
            1,
            hidden_dim,
        )

    def _normalize_input(self, features: torch.Tensor) -> torch.Tensor:
        """对进入 Adapter 的冻结 VLP 特征做可选预归一化。"""

        if self.input_normalization == "layernorm_l2":
            features = F.layer_norm(features, (features.shape[-1],))
        if self.input_normalization in {"l2", "layernorm_l2"}:
            features = F.normalize(features, dim=-1)
        return features

    def _bounded_radius(self, radius_logits: torch.Tensor) -> torch.Tensor:
        """将 radius head 输出限制到稳定半径范围。"""

        return self.radius_min + torch.sigmoid(radius_logits) * (self.radius_max - self.radius_min)

    def _map_features(self, features: torch.Tensor, direction_head: nn.Module, radius_head: nn.Module):
        """输出 direction、radius、tangent 和 Poincare ball 特征。"""

        features = self._normalize_input(features)
        direction = F.normalize(direction_head(features), dim=-1)
        radius = self._bounded_radius(radius_head(features))
        tangent = radius * direction
        ball_features = self.ball.project(self.ball.exp_map_zero(tangent))
        return direction, radius, tangent, ball_features

    def forward(
        self,
        visual_tokens_by_layer: Mapping[str, torch.Tensor],
        txt_embeds: torch.Tensor,
    ):
        """返回视觉/文本切空间特征与 Poincare ball 特征。"""

        unknown_layers = [name for name in visual_tokens_by_layer if name not in self.layer_name_to_key]
        if unknown_layers:
            raise KeyError(
                "Visual token layers are not supported by adapter checkpoint: "
                f"{unknown_layers}. Available layers: {self.layer_names}"
            )
        active_layer_names = [name for name in self.layer_names if name in visual_tokens_by_layer]
        if not active_layer_names:
            raise KeyError(
                "No visual token layers matched adapter checkpoint. "
                f"Available layers: {self.layer_names}"
            )

        visual_tangent_by_layer = OrderedDict()
        visual_h_by_layer = OrderedDict()
        visual_radius_by_layer = OrderedDict()
        for layer_name in active_layer_names:
            key = self.layer_name_to_key[layer_name]
            _, radius, tangent, ball_features = self._map_features(
                visual_tokens_by_layer[layer_name],
                self.visual_direction_heads[key],
                self.visual_radius_heads[key],
            )
            visual_tangent_by_layer[layer_name] = tangent
            visual_h_by_layer[layer_name] = ball_features
            visual_radius_by_layer[layer_name] = radius

        _, text_radius, text_tangent, text_h = self._map_features(
            txt_embeds,
            self.text_direction_head,
            self.text_radius_head,
        )
        return visual_tangent_by_layer, text_tangent, visual_h_by_layer, text_h, visual_radius_by_layer, text_radius

    def checkpoint_config(self) -> Dict:
        """生成 checkpoint 中用于重建 Adapter 的结构配置。"""

        return asdict(self.config)


def save_adapter_checkpoint(
    adapter: LearnableHyperbolicAdapter,
    checkpoint_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """保存 Adapter 权重和重建所需元数据。"""

    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_config": adapter.checkpoint_config(),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)


def load_adapter_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    freeze: bool = True,
) -> LearnableHyperbolicAdapter:
    """从 checkpoint 加载 `LearnableHyperbolicAdapter`。"""

    payload = torch.load(checkpoint_path, map_location=device or "cpu")
    if "adapter_config" not in payload or "adapter_state_dict" not in payload:
        raise ValueError("Adapter checkpoint must contain adapter_config and adapter_state_dict.")
    config = LearnableHyperbolicAdapterConfig(**payload["adapter_config"])
    adapter = LearnableHyperbolicAdapter(config)
    adapter.load_state_dict(payload["adapter_state_dict"])
    if device is not None:
        adapter.to(device)
    if freeze:
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
    return adapter
