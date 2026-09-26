"""庞加莱球空间的双曲图像侧攻击损失。"""

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hyperbolic import PoincareBall

from .hyperbolic_adapter import LearnableHyperbolicAdapter, load_adapter_checkpoint


@dataclass
class HyperbolicGuidanceConfig:
    """双曲空间引导配置，覆盖 Adapter 训练和攻击实施公共参数。"""

    lambda_hyp: float = 0.03
    adapter_checkpoint: Optional[str] = None
    adapter_dim: int = 256
    adapter_input_normalization: str = "layernorm_l2"
    adapter_train_data: str = "mscoco_train"
    adapter_train_objective: str = "clean_ranking_logmeanexp"
    adapter_freeze_in_attack: bool = True
    radius_min: float = 0.1
    radius_max: float = 0.8
    curvature: float = 1.0
    eps: float = 1e-5
    alignment_temperature: float = 0.1
    radius_temperature: float = 0.1
    layer_weight_strategy: str = "fixed_depth"
    layer_weights: Optional[Sequence[float]] = None
    hard_negative_topk: int = 5
    ranking_temperature: float = 0.1
    attack_loss_mode: str = "positive_distance"
    gradient_fusion_strategy: str = "pcgrad_norm"
    geometry_mode: str = "hyperbolic"
    use_radius_gap: bool = True


class HyperbolicGuidanceLoss(nn.Module):
    """基于庞加莱球加权双曲距离构造可切换的 attack loss。"""

    def __init__(
        self,
        config: Optional[HyperbolicGuidanceConfig] = None,
        adapter: Optional[LearnableHyperbolicAdapter] = None,
        **legacy_kwargs,
    ):
        super().__init__()
        config = self._build_config(config, legacy_kwargs)
        self._validate_config(config)

        self.config = config
        self.lambda_hyp = float(config.lambda_hyp)
        self.alignment_temperature = float(config.alignment_temperature)
        self.radius_temperature = float(config.radius_temperature)
        self.layer_weight_strategy = config.layer_weight_strategy
        self.layer_weights = list(config.layer_weights) if config.layer_weights is not None else None
        self.hard_negative_topk = int(config.hard_negative_topk)
        self.ranking_temperature = float(config.ranking_temperature)
        self.ball = PoincareBall(curvature=config.curvature, eps=config.eps)

        self.adapter = adapter
        if self.adapter is None and config.adapter_checkpoint:
            self.adapter = load_adapter_checkpoint(
                config.adapter_checkpoint,
                freeze=config.adapter_freeze_in_attack,
            )

    def _build_config(
        self,
        config: Optional[HyperbolicGuidanceConfig],
        legacy_kwargs: Dict,
    ) -> HyperbolicGuidanceConfig:
        """兼容少量旧关键字；主线配置只使用 learnable Adapter 字段。"""

        if config is not None and not isinstance(config, HyperbolicGuidanceConfig):
            raise TypeError("config must be a HyperbolicGuidanceConfig instance.")
        config = config or HyperbolicGuidanceConfig()
        if not legacy_kwargs:
            return config

        aliases = {
            "hyperbolic_curvature": "curvature",
            "hyperbolic_eps": "eps",
        }
        config_fields = {item.name for item in fields(HyperbolicGuidanceConfig)}
        updates = {}
        for key, value in legacy_kwargs.items():
            if key == "hyperbolic_tau":
                updates["alignment_temperature"] = value
                updates["radius_temperature"] = value
                continue
            key = aliases.get(key, key)
            if key not in config_fields:
                raise TypeError(f"Unsupported HyperbolicGuidanceLoss argument: {key}")
            updates[key] = value
        if "alignment_temperature" in updates and "radius_temperature" not in updates:
            updates["radius_temperature"] = updates["alignment_temperature"]
        return replace(config, **updates)

    def _validate_config(self, config: HyperbolicGuidanceConfig) -> None:
        """尽早暴露不可用实验参数。"""

        if config.lambda_hyp < 0:
            raise ValueError("lambda_hyp must be non-negative.")
        if config.adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive.")
        if config.adapter_input_normalization not in {"none", "l2", "layernorm_l2"}:
            raise ValueError(f"Unsupported adapter_input_normalization: {config.adapter_input_normalization}")
        if config.adapter_train_data not in {"mscoco_train", "flickr30k_train", "imagenet_templates"}:
            raise ValueError(f"Unsupported adapter_train_data: {config.adapter_train_data}")
        if config.adapter_train_objective != "clean_ranking_logmeanexp":
            raise ValueError(f"Unsupported adapter_train_objective: {config.adapter_train_objective}")
        if config.radius_min <= 0 or config.radius_min >= config.radius_max:
            raise ValueError("radius_min and radius_max must satisfy 0 < min < max.")
        if config.curvature <= 0:
            raise ValueError("curvature must be positive.")
        if config.alignment_temperature <= 0:
            raise ValueError("alignment_temperature must be positive.")
        if config.radius_temperature <= 0:
            raise ValueError("radius_temperature must be positive.")
        if config.layer_weight_strategy not in {"uniform", "fixed_depth", "late_only"}:
            raise ValueError(f"Unsupported layer_weight_strategy: {config.layer_weight_strategy}")
        if config.hard_negative_topk <= 0:
            raise ValueError("hard_negative_topk must be positive.")
        if config.ranking_temperature <= 0:
            raise ValueError("ranking_temperature must be positive.")
        if config.attack_loss_mode not in {"ranking", "positive_distance", "hybrid"}:
            raise ValueError(f"Unsupported attack_loss_mode: {config.attack_loss_mode}")
        if config.gradient_fusion_strategy not in {
            "additive",
            "normalized_additive",
            "pcgrad_norm",
            "candidate_guided",
        }:
            raise ValueError(f"Unsupported gradient_fusion_strategy: {config.gradient_fusion_strategy}")
        if config.geometry_mode not in {"hyperbolic", "euclidean"}:
            raise ValueError(f"Unsupported geometry_mode: {config.geometry_mode}")

    def set_adapter(self, adapter: LearnableHyperbolicAdapter) -> None:
        """训练脚本可显式注入正在训练的 Adapter。"""

        self.adapter = adapter

    def project_features_to_hyperbolic(
        self,
        visual_tokens_by_layer: Dict[str, torch.Tensor],
        txt_embeds: torch.Tensor,
    ):
        """通过 Adapter 返回切空间特征和 Poincare ball 特征。"""

        if self.adapter is None:
            raise RuntimeError(
                "LearnableHyperbolicAdapter is required. Provide adapter during training "
                "or set adapter_checkpoint for attack inference."
            )
        self.adapter.to(txt_embeds.device)
        (
            visual_tangent_by_layer,
            text_tangent,
            visual_h_by_layer,
            text_h,
            _,
            _,
        ) = self.adapter(visual_tokens_by_layer, txt_embeds)
        return visual_tangent_by_layer, text_tangent, visual_h_by_layer, text_h

    def map_to_poincare(self, tangent_features: torch.Tensor) -> torch.Tensor:
        """执行 `expmap0` 与 `project_to_ball`。"""

        return self.ball.project(self.ball.exp_map_zero(tangent_features))

    def compute_relation_matrix(
        self,
        visual_tangent_by_layer: Dict[str, torch.Tensor],
        text_tangent: torch.Tensor,
        visual_h_by_layer: Dict[str, torch.Tensor],
        text_h: torch.Tensor,
    ) -> torch.Tensor:
        """计算并聚合为层-caption 双曲关系矩阵 `[B,L,M]`。"""

        if self.config.geometry_mode == "euclidean":
            _, _, _, relation_matrix = self.compute_alignment_weights(visual_tangent_by_layer, text_tangent)
        else:
            _, _, _, relation_matrix = self.compute_alignment_weights(visual_h_by_layer, text_h)
        return relation_matrix

    def _stack_layer_tensors(
        self,
        tensors_by_layer: Dict[str, torch.Tensor],
        tensor_name: str,
    ) -> Tuple[Tuple[str, ...], torch.Tensor]:
        """将多层 `[B,N,D]` tensor 堆叠为 `[B,L,N,D]`。

        v1 主方法需要显式的 `[B,L,N,M]` 对齐权重，因此不同层 token
        数不一致时直接报错，避免静默 padding 改变对齐语义。
        """

        if not tensors_by_layer:
            raise ValueError(f"{tensor_name} must not be empty.")

        layer_names = tuple(tensors_by_layer.keys())
        reference_shape = None
        tensors = []
        for layer_name in layer_names:
            tensor = tensors_by_layer[layer_name]
            if tensor.dim() != 3:
                raise ValueError(
                    f"{tensor_name}[{layer_name}] must have shape [B,N,D], "
                    f"got {tuple(tensor.shape)}."
                )
            shape_key = tuple(tensor.shape[:2])
            if reference_shape is None:
                reference_shape = shape_key
            elif shape_key != reference_shape:
                raise ValueError(
                    f"{tensor_name} layers must share [B,N] for A_hyp. "
                    f"Expected {reference_shape}, got {shape_key} at {layer_name}."
                )
            tensors.append(tensor)
        return layer_names, torch.stack(tensors, dim=1)

    def compute_alignment_weights(
        self,
        visual_h_by_layer: Dict[str, torch.Tensor],
        text_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 `A_hyp`、token-caption 双曲距离、径向差和层级统计。"""

        _layer_names, visual_features = self._stack_layer_tensors(visual_h_by_layer, "visual_h_by_layer")
        if text_h.dim() != 2:
            raise ValueError(f"text_h must have shape [M,D], got {tuple(text_h.shape)}.")
        if visual_features.shape[-1] != text_h.shape[-1]:
            raise ValueError(
                "visual_h_by_layer and text_h must share adapter_dim, "
                f"got {visual_features.shape[-1]} and {text_h.shape[-1]}."
            )

        text_h_expanded = text_h.view(1, 1, 1, text_h.shape[0], text_h.shape[-1])
        if self.config.geometry_mode == "euclidean":
            distances = torch.linalg.vector_norm(visual_features.unsqueeze(3) - text_h_expanded, dim=-1)
            visual_radius = torch.linalg.vector_norm(visual_features, dim=-1).unsqueeze(-1)
            text_radius = torch.linalg.vector_norm(text_h, dim=-1).view(1, 1, 1, text_h.shape[0])
        else:
            distances = self.ball.distance(visual_features.unsqueeze(3), text_h_expanded)
            visual_radius = self.ball.norm(visual_features).unsqueeze(-1)
            text_radius = self.ball.norm(text_h).view(1, 1, 1, text_h.shape[0])
        radius_gap = torch.abs(visual_radius - text_radius)

        logits = -distances / self.alignment_temperature
        if self.config.use_radius_gap:
            logits = logits - radius_gap / self.radius_temperature
        # Each visual layer represents a separate semantic level.  Normalize
        # tokens inside a layer so that a layer's relation distance is not
        # implicitly scaled by how much probability mass it won from the
        # other layers.
        alignment_weights = torch.softmax(logits, dim=2)

        relation_matrix = (alignment_weights * distances).sum(dim=2)
        return alignment_weights, distances, radius_gap, relation_matrix

    def compute_hyperbolic_scores(
        self,
        alignment_weights: torch.Tensor,
        distances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """聚合庞加莱球双曲距离并返回 `hyperbolic_scores [B,M]`。"""

        if alignment_weights.shape != distances.shape:
            raise ValueError(
                "alignment_weights and distances must have the same shape, "
                f"got {tuple(alignment_weights.shape)} and {tuple(distances.shape)}."
            )

        per_layer_distances = (alignment_weights * distances).sum(dim=2)
        weighted_hyperbolic_distances = self.aggregate_layer_distances(per_layer_distances)
        hyperbolic_scores = -weighted_hyperbolic_distances
        return weighted_hyperbolic_distances, hyperbolic_scores

    def aggregate_layer_distances(self, relation_matrix: torch.Tensor) -> torch.Tensor:
        """Aggregate `[B,L,M]` distances with normalized semantic-layer weights."""

        if relation_matrix.dim() != 3:
            raise ValueError("relation_matrix must have shape [B,L,M].")
        layer_weights = self.compute_layer_weights(
            relation_matrix.shape[1],
            relation_matrix.device,
            relation_matrix.dtype,
        )
        return (relation_matrix * layer_weights.view(1, -1, 1)).sum(dim=1)

    def build_caption_masks(
        self,
        txt2img,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造 batch 内 positive / negative caption mask。"""

        txt2img_tensor = torch.as_tensor(txt2img, device=device, dtype=torch.long)
        num_texts = int(txt2img_tensor.numel())
        image_indices = torch.arange(batch_size, device=device).view(batch_size, 1)
        caption_image_indices = txt2img_tensor.view(1, num_texts)
        valid_caption = (caption_image_indices >= 0) & (caption_image_indices < batch_size)
        positive_mask = (caption_image_indices == image_indices) & valid_caption
        negative_mask = (caption_image_indices != image_indices) & valid_caption
        return positive_mask.to(dtype=dtype), negative_mask.to(dtype=dtype)

    def build_positive_caption_mask(
        self,
        txt2img,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """兼容旧调用，返回 positive caption mask。"""

        positive_mask, _ = self.build_caption_masks(txt2img, batch_size, device, dtype)
        return positive_mask

    def compute_layer_weights(
        self,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成归一化层权重。"""

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
            weights = default if num_layers == 4 else torch.linspace(1.0, 2.0, steps=num_layers, device=device, dtype=dtype)
        return weights / weights.sum().clamp_min(1e-12)

    def _masked_logmeanexp(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """在 caption 维度上执行 masked logmeanexp。"""

        neg_large = torch.finfo(scores.dtype).min
        expanded_mask = mask.bool()
        if scores.dim() == 3:
            expanded_mask = expanded_mask.unsqueeze(1)
        elif scores.dim() != 2:
            raise ValueError(f"scores must have shape [B,M] or [B,L,M], got {tuple(scores.shape)}.")
        counts = expanded_mask.sum(dim=-1).clamp_min(1)
        return scores.masked_fill(~expanded_mask, neg_large).logsumexp(dim=-1) - counts.to(scores.dtype).log()

    def _topk_negative_logmeanexp(
        self,
        scores: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        """仅保留 batch 内 top-k hard negatives 后计算 logmeanexp。"""

        neg_large = torch.finfo(scores.dtype).min
        expanded_mask = negative_mask.bool()
        if scores.dim() == 3:
            expanded_mask = expanded_mask.unsqueeze(1)
        elif scores.dim() != 2:
            raise ValueError(f"scores must have shape [B,M] or [B,L,M], got {tuple(scores.shape)}.")
        masked_scores = scores.masked_fill(~expanded_mask, neg_large)
        k = min(self.hard_negative_topk, masked_scores.shape[-1])
        topk_scores = masked_scores.topk(k=k, dim=-1).values
        valid_topk = topk_scores > (neg_large / 2)
        counts = valid_topk.sum(dim=-1).clamp_min(1)
        return topk_scores.masked_fill(~valid_topk, neg_large).logsumexp(dim=-1) - counts.to(scores.dtype).log()

    def compute_ranking_loss(
        self,
        relation_matrix: torch.Tensor,
        txt2img,
    ) -> torch.Tensor:
        """Adapter 离线训练使用的 clean 双曲距离 ranking 目标。"""

        if relation_matrix.dim() != 3:
            raise ValueError("relation_matrix must have shape [B,L,M].")
        batch_size, _, _ = relation_matrix.shape
        device = relation_matrix.device
        dtype = relation_matrix.dtype
        positive_mask, negative_mask = self.build_caption_masks(txt2img, batch_size, device, dtype)
        valid_images = (
            (positive_mask.sum(dim=1) > 0)
            & (negative_mask.sum(dim=1) > 0)
        ).to(dtype=dtype)
        valid_count = valid_images.sum()
        if valid_count.item() == 0:
            return relation_matrix.sum() * 0.0

        aggregated_distances = self.aggregate_layer_distances(relation_matrix)
        scores = -aggregated_distances / self.ranking_temperature
        positive_scores = self._masked_logmeanexp(scores, positive_mask)
        negative_scores = self._topk_negative_logmeanexp(scores, negative_mask)

        per_image_loss = negative_scores - positive_scores
        return (per_image_loss * valid_images).sum() / valid_count

    def compute_hyperbolic_ball_ranking_loss(
        self,
        hyperbolic_scores: torch.Tensor,
        txt2img,
    ) -> torch.Tensor:
        """计算 batch hard-negative 形式的 `L_hyp_ball_rank`。"""

        if hyperbolic_scores.dim() != 2:
            raise ValueError(
                "hyperbolic_scores must have shape [B,M], "
                f"got {tuple(hyperbolic_scores.shape)}."
            )
        batch_size = hyperbolic_scores.shape[0]
        device = hyperbolic_scores.device
        dtype = hyperbolic_scores.dtype
        positive_mask, negative_mask = self.build_caption_masks(txt2img, batch_size, device, dtype)
        valid_images = (
            (positive_mask.sum(dim=1) > 0)
            & (negative_mask.sum(dim=1) > 0)
        ).to(dtype=dtype)
        valid_count = valid_images.sum()
        if valid_count.item() == 0:
            return hyperbolic_scores.sum() * 0.0

        scores = hyperbolic_scores / self.ranking_temperature
        positive_scores = self._masked_logmeanexp(scores, positive_mask)
        negative_scores = self._topk_negative_logmeanexp(scores, negative_mask)
        per_image_loss = negative_scores - positive_scores
        return (per_image_loss * valid_images).sum() / valid_count

    def compute_positive_distance_attack_loss(
        self,
        weighted_hyperbolic_distances: torch.Tensor,
        txt2img,
    ) -> torch.Tensor:
        """TEDFusion 式正样本双曲距离攻击项。

        TEDFusion 在训练中最小化径向加权图文双曲距离以促进语义对齐；
        迁移攻击阶段反向最大化正 caption 的加权双曲距离，从而优先破坏
        跨模型共享的图文匹配语义，而不是过度依赖源模型 hard negative。
        """

        if weighted_hyperbolic_distances.dim() != 2:
            raise ValueError(
                "weighted_hyperbolic_distances must have shape [B,M], "
                f"got {tuple(weighted_hyperbolic_distances.shape)}."
            )
        batch_size = weighted_hyperbolic_distances.shape[0]
        device = weighted_hyperbolic_distances.device
        dtype = weighted_hyperbolic_distances.dtype
        positive_mask, _ = self.build_caption_masks(txt2img, batch_size, device, dtype)
        valid_images = (positive_mask.sum(dim=1) > 0).to(dtype=dtype)
        valid_count = valid_images.sum()
        if valid_count.item() == 0:
            return weighted_hyperbolic_distances.sum() * 0.0

        positive_counts = positive_mask.sum(dim=1).clamp_min(1.0)
        per_image_loss = (weighted_hyperbolic_distances * positive_mask).sum(dim=1) / positive_counts
        return (per_image_loss * valid_images).sum() / valid_count

    def compute_hyperbolic_attack_loss(
        self,
        hyperbolic_scores: torch.Tensor,
        weighted_hyperbolic_distances: torch.Tensor,
        txt2img,
    ) -> torch.Tensor:
        """按配置选择攻击阶段双曲目标。"""

        if self.config.attack_loss_mode == "ranking":
            return self.compute_hyperbolic_ball_ranking_loss(hyperbolic_scores, txt2img)
        positive_distance_loss = self.compute_positive_distance_attack_loss(
            weighted_hyperbolic_distances,
            txt2img,
        )
        if self.config.attack_loss_mode == "positive_distance":
            return positive_distance_loss
        ranking_loss = self.compute_hyperbolic_ball_ranking_loss(hyperbolic_scores, txt2img)
        return positive_distance_loss + ranking_loss

    def compute_adapter_training_loss(
        self,
        visual_tokens_by_layer: Dict[str, torch.Tensor],
        txt_embeds: torch.Tensor,
        txt2img,
    ):
        """Adapter 离线训练目标：只优化 clean 双曲距离 ranking。"""

        visual_tangent_by_layer, text_tangent, visual_h_by_layer, text_h = self.project_features_to_hyperbolic(
            visual_tokens_by_layer,
            txt_embeds,
        )
        relation_matrix = self.compute_relation_matrix(
            visual_tangent_by_layer,
            text_tangent,
            visual_h_by_layer,
            text_h,
        )
        loss = self.compute_ranking_loss(relation_matrix, txt2img)
        stats = self.compute_stats(relation_matrix, txt2img)
        stats["loss_role"] = "adapter_clean_hyperbolic_ranking"
        return loss, relation_matrix, stats

    def compute_stats(self, relation_matrix: torch.Tensor, txt2img) -> Dict[str, float]:
        """Adapter clean 双曲距离 ranking 的正负 caption 统计。"""

        batch_size, _, _ = relation_matrix.shape
        positive_mask, negative_mask = self.build_caption_masks(
            txt2img,
            batch_size,
            relation_matrix.device,
            relation_matrix.dtype,
        )
        pos_mask = positive_mask.bool()
        neg_mask = negative_mask.bool()
        aggregated_distances = self.aggregate_layer_distances(relation_matrix)
        stats = {
            "ranking_temperature": self.ranking_temperature,
            "hard_negative_topk": self.hard_negative_topk,
            "valid_ranking_images": int(
                ((positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)).sum().detach().item()
            ),
        }
        with torch.no_grad():
            if pos_mask.any():
                positive_distance = aggregated_distances.masked_select(pos_mask)
                stats["positive_caption_distance_mean"] = float(positive_distance.mean().detach().cpu())
                stats["positive_caption_score_mean"] = float((-positive_distance).mean().detach().cpu())
            if neg_mask.any():
                negative_distance = aggregated_distances.masked_select(neg_mask)
                stats["negative_caption_distance_mean"] = float(negative_distance.mean().detach().cpu())
                stats["negative_caption_score_mean"] = float((-negative_distance).mean().detach().cpu())
            if "positive_caption_score_mean" in stats and "negative_caption_score_mean" in stats:
                stats["ranking_margin_mean"] = stats["negative_caption_score_mean"] - stats["positive_caption_score_mean"]
        return stats

    def compute_relation_stats(self, relation_matrix: torch.Tensor, txt2img) -> Dict[str, float]:
        """兼容旧调用；主路径请使用 `compute_stats()`。"""

        return self.compute_stats(relation_matrix, txt2img)

    def compute_ball_stats(
        self,
        hyperbolic_scores: torch.Tensor,
        weighted_hyperbolic_distances: torch.Tensor,
        relation_matrix: torch.Tensor,
        alignment_weights: torch.Tensor,
        radius_gap: torch.Tensor,
        txt2img,
    ) -> Dict[str, float]:
        """输出主线庞加莱球双曲 score 与对齐权重统计。"""

        batch_size = hyperbolic_scores.shape[0]
        positive_mask, negative_mask = self.build_caption_masks(
            txt2img,
            batch_size,
            hyperbolic_scores.device,
            hyperbolic_scores.dtype,
        )
        stats = {
            "loss_role": "hyperbolic_ball_ranking",
            "geometry_mode": self.config.geometry_mode,
            "use_radius_gap": self.config.use_radius_gap,
            "alignment_temperature": self.alignment_temperature,
            "radius_temperature": self.radius_temperature,
            "ranking_temperature": self.ranking_temperature,
            "hard_negative_topk": self.hard_negative_topk,
            "relation_matrix_shape": tuple(relation_matrix.shape),
            "alignment_weights_shape": tuple(alignment_weights.shape),
            "weighted_hyperbolic_distances_shape": tuple(weighted_hyperbolic_distances.shape),
            "hyperbolic_scores_shape": tuple(hyperbolic_scores.shape),
            "radius_gap_shape": tuple(radius_gap.shape),
            "valid_ranking_images": int(
                ((positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)).sum().detach().item()
            ),
        }
        with torch.no_grad():
            stats["mean_hyp_distance"] = float(relation_matrix.detach().mean().cpu())
            stats["weighted_hyperbolic_distance_mean"] = float(
                weighted_hyperbolic_distances.detach().mean().cpu()
            )
            stats["radius_gap_mean"] = float(radius_gap.detach().mean().cpu())
            stats["radius_gap_max"] = float(radius_gap.detach().max().cpu())
            stats["alignment_weight_max_mean"] = float(
                alignment_weights.detach().flatten(start_dim=1, end_dim=2).max(dim=1).values.mean().cpu()
            )
            if positive_mask.bool().any():
                positive_scores = hyperbolic_scores.masked_select(positive_mask.bool())
                stats["positive_caption_score_mean"] = float(positive_scores.mean().detach().cpu())
            if negative_mask.bool().any():
                negative_scores = hyperbolic_scores.masked_select(negative_mask.bool())
                stats["negative_caption_score_mean"] = float(negative_scores.mean().detach().cpu())
            if "positive_caption_score_mean" in stats and "negative_caption_score_mean" in stats:
                stats["ranking_margin_mean"] = (
                    stats["negative_caption_score_mean"] - stats["positive_caption_score_mean"]
                )
        return stats

    def hyperbolic_guidance_loss(
        self,
        visual_tokens_by_layer: Dict[str, torch.Tensor],
        txt_embeds: torch.Tensor,
        txt2img,
    ):
        """串联 Adapter、双曲距离权重和庞加莱球 ranking loss。"""

        visual_tangent_by_layer, text_tangent, visual_h_by_layer, text_h = self.project_features_to_hyperbolic(
            visual_tokens_by_layer,
            txt_embeds,
        )
        if self.config.geometry_mode == "euclidean":
            distance_visual_by_layer = visual_tangent_by_layer
            distance_text = text_tangent
        else:
            distance_visual_by_layer = visual_h_by_layer
            distance_text = text_h
        alignment_weights, distances, radius_gap, relation_matrix = self.compute_alignment_weights(
            distance_visual_by_layer,
            distance_text,
        )
        weighted_hyperbolic_distances, hyperbolic_scores = self.compute_hyperbolic_scores(
            alignment_weights,
            distances,
        )
        loss = self.compute_hyperbolic_attack_loss(
            hyperbolic_scores,
            weighted_hyperbolic_distances,
            txt2img,
        )
        ball_stats = self.compute_ball_stats(
            hyperbolic_scores,
            weighted_hyperbolic_distances,
            relation_matrix,
            alignment_weights,
            radius_gap,
            txt2img,
        )
        if self.config.geometry_mode == "euclidean":
            visual_radii = torch.cat(
                [
                    torch.linalg.vector_norm(visual_h, dim=-1).detach().reshape(-1)
                    for visual_h in visual_tangent_by_layer.values()
                ],
                dim=0,
            )
            text_radii = torch.linalg.vector_norm(text_tangent, dim=-1).detach()
        else:
            visual_radii = torch.cat(
                [self.ball.norm(visual_h).detach().reshape(-1) for visual_h in visual_h_by_layer.values()],
                dim=0,
            )
            text_radii = self.ball.norm(text_h).detach()
        stats = {
            "mean_hyp_distance": float(relation_matrix.detach().mean().cpu()),
            "geometry_mode": self.config.geometry_mode,
            "use_radius_gap": self.config.use_radius_gap,
            "mean_visual_radius": float(visual_radii.mean().cpu()),
            "min_visual_radius": float(visual_radii.min().cpu()),
            "max_visual_radius": float(visual_radii.max().cpu()),
            "mean_text_radius": float(text_radii.mean().cpu()),
            "min_text_radius": float(text_radii.min().cpu()),
            "max_text_radius": float(text_radii.max().cpu()),
            "attack_loss_mode": self.config.attack_loss_mode,
            "gradient_fusion_strategy": self.config.gradient_fusion_strategy,
        }
        stats.update(ball_stats)
        outputs = {
            "alignment_weights": alignment_weights,
            "relation_matrix": relation_matrix,
            "hyperbolic_distances": distances,
            "radius_gap": radius_gap,
            "weighted_hyperbolic_distances": weighted_hyperbolic_distances,
            "hyperbolic_scores": hyperbolic_scores,
            "visual_tangent_by_layer": visual_tangent_by_layer,
            "text_tangent": text_tangent,
        }
        return loss, outputs, stats

    def forward(
        self,
        visual_tokens_by_layer: Dict[str, torch.Tensor],
        txt_embeds: torch.Tensor,
        txt2img,
    ):
        """返回未加权的双曲 attack loss、outputs 和统计。"""

        return self.hyperbolic_guidance_loss(visual_tokens_by_layer, txt_embeds, txt2img)

    def weighted_loss(
        self,
        visual_tokens_by_layer: Dict[str, torch.Tensor],
        txt_embeds: torch.Tensor,
        txt2img,
    ) -> torch.Tensor:
        """返回加权后的双曲 attack loss。"""

        loss, _, _ = self.forward(visual_tokens_by_layer, txt_embeds, txt2img)
        return self.lambda_hyp * loss
