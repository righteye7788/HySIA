import numpy as np
import torch
import torch.nn as nn

import copy
from dataclasses import dataclass
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
import time

from hierarchical_visual_attack import (
    HyperbolicGuidanceConfig,
    HyperbolicGuidanceLoss,
    HierarchicalVisionExtractor,
)

feature_layers = [
    "transformer.resblocks.2",   # 浅层
    "transformer.resblocks.5",   # 中浅层
    "transformer.resblocks.8",   # 中深层
    "transformer.resblocks.11",  # 最深层
]


@dataclass
class ImageAttackObjectiveConfig:
    """图像侧攻击目标路由配置。"""

    attack_objective: str = "original_plus_hyperbolic"
    lambda_hyp: float = 0.03

    def __post_init__(self):
        choices = {"baseline", "original_plus_hyperbolic", "hyperbolic_main"}
        if self.attack_objective not in choices:
            raise ValueError(f"Unsupported attack_objective: {self.attack_objective}")
        if self.lambda_hyp < 0:
            raise ValueError("lambda_hyp must be non-negative.")


class Attacker():
    def __init__(
        self,
        model,
        img_attacker,
        txt_attacker,
        feature_layers=None,
        token_mode="patch",
        sequence_first=True,
        cnn_token_grid_size=None,
    ):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.feature_layers = feature_layers or globals()["feature_layers"]
        self.sequence_first = sequence_first
        self.cnn_token_grid_size = cnn_token_grid_size
        self.vision_extractor = None
        if self.img_attacker.uses_multilevel_features:
            self.vision_extractor = HierarchicalVisionExtractor(
                model=self.model,
                feature_layers=self.feature_layers,
                token_mode=token_mode,
                sequence_first=self.sequence_first,
                cnn_token_grid_size=self.cnn_token_grid_size,
            )

    def attack(self, imgs, txts, txt2img, all_texts,device='cpu', max_length=30, scales=None, masks=None, **kwargs):
        # 基本流程
        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']

            all_texts_input = self.txt_attacker.tokenizer(all_texts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            all_texts_output = self.model.inference_text(all_texts_input)
            all_txt_supervisions = all_texts_output['text_feat']


        start_time = time.time()
        adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img,all_txt_supervisions, device, self.vision_extractor,
                                                                      scales=scales, txt_embeds=txt_supervisions)
        end_time = time.time()
        execuate_time = end_time - start_time

        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            last_adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(last_adv_imgs))
            last_adv_img_supervisions = last_adv_imgs_outputs['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
                                                       adv_img_embeds=adv_img_supervisions,
                                                       last_adv_img_embeds=last_adv_img_supervisions)
        return adv_imgs, adv_txts, execuate_time

    # def attack(self, imgs, txts, txt2img, all_texts,device='cpu', max_length=30, scales=None, masks=None, **kwargs):
    #     # 更改为【图-文-图】的攻击流程，避免文本锚点时刻动荡导致的攻击失败

    #     # 图像攻击
    #     with torch.no_grad():
    #         txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True,
    #                                                  max_length=max_length, return_tensors="pt").to(device)
    #         txts_output = self.model.inference_text(txts_input)
    #         txt_supervisions = txts_output['text_feat']

    #         all_texts_input = self.txt_attacker.tokenizer(all_texts, padding='max_length', truncation=True,
    #                                                  max_length=max_length, return_tensors="pt").to(device)
    #         all_texts_output = self.model.inference_text(all_texts_input)
    #         all_txt_supervisions = all_texts_output['text_feat']


    #     start_time = time.time()
    #     adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img,all_txt_supervisions, device, self.vision_extractor,
    #                                                                   scales=scales, txt_embeds=txt_supervisions)
    #     end_time = time.time()
    #     execuate_time = end_time - start_time

    #     # 文本攻击
    #     with torch.no_grad():
    #         origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
    #         img_supervisions = origin_img_output['image_feat'][txt2img]
    #         adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
    #         adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
    #         last_adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(last_adv_imgs))
    #         last_adv_img_supervisions = last_adv_imgs_outputs['image_feat'][txt2img]
    #     adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
    #                                                    adv_img_embeds=adv_img_supervisions,
    #                                                    last_adv_img_embeds=last_adv_img_supervisions)

    #     # # 图像攻击
    #     # with torch.no_grad():
    #     #     txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True,
    #     #                                              max_length=max_length, return_tensors="pt").to(device)
    #     #     txts_output = self.model.inference_text(txts_input)
    #     #     txt_supervisions = txts_output['text_feat']

    #     #     all_texts_input = self.txt_attacker.tokenizer(all_texts, padding='max_length', truncation=True,
    #     #                                              max_length=max_length, return_tensors="pt").to(device)
    #     #     all_texts_output = self.model.inference_text(all_texts_input)
    #     #     all_txt_supervisions = all_texts_output['text_feat']


    #     # start_time = time.time()
    #     # adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img,all_txt_supervisions, device, self.vision_extractor,
    #     #                                                               scales=scales, txt_embeds=txt_supervisions)
    #     # end_time = time.time()
    #     # execuate_time = end_time - start_time

    #     return adv_imgs, adv_txts, execuate_time


class ImageAttacker():
    def __init__(
        self,
        normalization,
        eps=2 / 255,
        steps=10,
        step_size=0.5 / 255,
        sample_numbers=5,
        attack_objective_config=None,
        hyperbolic_config=None,
        attack_objective=None,
        lambda_hyp=None,
        **legacy_kwargs,
    ):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers
        self.latest_hyperbolic_stats = {}
        self.latest_image_loss_stats = {}
        self.objective_config = self._build_objective_config(
            attack_objective_config,
            attack_objective,
            lambda_hyp,
            legacy_kwargs,
        )
        self.hyperbolic_loss = HyperbolicGuidanceLoss(
            self._build_hyperbolic_config(hyperbolic_config, self.objective_config, legacy_kwargs)
        )

    def _build_objective_config(
        self,
        attack_objective_config,
        attack_objective,
        lambda_hyp,
        legacy_kwargs,
    ):
        """构造新攻击目标配置，并兼容旧双曲开关。"""

        if attack_objective_config is not None:
            if not isinstance(attack_objective_config, ImageAttackObjectiveConfig):
                raise TypeError("attack_objective_config must be ImageAttackObjectiveConfig.")
            return attack_objective_config

        resolved_objective = attack_objective
        if resolved_objective is None:
            resolved_objective = self._resolve_attack_objective_from_legacy(legacy_kwargs)
        resolved_lambda = (
            float(lambda_hyp)
            if lambda_hyp is not None
            else float(legacy_kwargs.get("lambda_hyp", 0.03))
        )
        return ImageAttackObjectiveConfig(
            attack_objective=resolved_objective,
            lambda_hyp=resolved_lambda,
        )

    def _resolve_attack_objective_from_legacy(self, legacy_kwargs):
        """把旧 CLI 参数映射到新的 `attack_objective`。"""

        auxiliary_loss_type = legacy_kwargs.get("auxiliary_loss_type")
        enable_hier_visual = legacy_kwargs.get("enable_hier_visual", False)
        enable_hyperbolic_guidance = legacy_kwargs.get("enable_hyperbolic_guidance", False)
        image_loss_composition = legacy_kwargs.get("image_loss_composition", "original_plus_aux")

        if auxiliary_loss_type == "euclidean_token_caption" or enable_hier_visual:
            raise ValueError(
                "euclidean_token_caption / enable_hier_visual is deprecated in the "
                "main method. Use attack_objective='baseline', "
                "'original_plus_hyperbolic', or 'hyperbolic_main'."
            )
        if auxiliary_loss_type == "hyperbolic_adapter" or enable_hyperbolic_guidance:
            if image_loss_composition == "hyperbolic_main":
                return "hyperbolic_main"
            return "original_plus_hyperbolic"
        return "baseline"

    def _build_hyperbolic_config(self, hyperbolic_config, objective_config, legacy_kwargs):
        """用配置对象收敛双曲参数，同时兼容旧关键字名称。"""

        if hyperbolic_config is not None:
            if not isinstance(hyperbolic_config, HyperbolicGuidanceConfig):
                raise TypeError("hyperbolic_config must be HyperbolicGuidanceConfig.")
            config = HyperbolicGuidanceConfig(
                **{
                    **hyperbolic_config.__dict__,
                    "lambda_hyp": objective_config.lambda_hyp,
                }
            )
            self._validate_adapter_checkpoint(objective_config, config)
            return config

        config = HyperbolicGuidanceConfig(
            lambda_hyp=objective_config.lambda_hyp,
            adapter_checkpoint=legacy_kwargs.get("adapter_checkpoint"),
            adapter_dim=legacy_kwargs.get("adapter_dim", 256),
            adapter_input_normalization=legacy_kwargs.get("adapter_input_normalization", "layernorm_l2"),
            adapter_train_data=legacy_kwargs.get("adapter_train_data", "mscoco_train"),
            adapter_train_objective=legacy_kwargs.get("adapter_train_objective", "clean_ranking_logmeanexp"),
            adapter_freeze_in_attack=legacy_kwargs.get("adapter_freeze_in_attack", True),
            radius_min=legacy_kwargs.get("radius_min", 0.1),
            radius_max=legacy_kwargs.get("radius_max", 0.8),
            curvature=legacy_kwargs.get("hyperbolic_curvature", 1.0),
            eps=legacy_kwargs.get("hyperbolic_eps", 1e-5),
            alignment_temperature=legacy_kwargs.get(
                "alignment_temperature",
                legacy_kwargs.get("hyperbolic_tau", 0.1),
            ),
            radius_temperature=legacy_kwargs.get(
                "radius_temperature",
                legacy_kwargs.get(
                    "alignment_temperature",
                    legacy_kwargs.get("hyperbolic_tau", 0.1),
                ),
            ),
            layer_weight_strategy=legacy_kwargs.get("layer_weight_strategy", "fixed_depth"),
            layer_weights=legacy_kwargs.get("layer_weights"),
            hard_negative_topk=legacy_kwargs.get("hard_negative_topk", 5),
            ranking_temperature=legacy_kwargs.get("ranking_temperature", 0.1),
            geometry_mode=legacy_kwargs.get("geometry_mode", "hyperbolic"),
            use_radius_gap=legacy_kwargs.get("use_radius_gap", True),
        )
        self._validate_adapter_checkpoint(objective_config, config)
        return config

    def _validate_adapter_checkpoint(self, objective_config, hyperbolic_config):
        """攻击实施阶段启用双曲目标时必须加载离线训练好的 Adapter。"""

        if (
            objective_config.attack_objective in {"original_plus_hyperbolic", "hyperbolic_main"}
            and not hyperbolic_config.adapter_checkpoint
        ):
            raise ValueError(
                "adapter_checkpoint is required for learnable hyperbolic attack objectives. "
                "Train LearnableHyperbolicAdapter first with scripts/train_hyperbolic_adapter.py."
            )

    @property
    def uses_multilevel_features(self):
        """判断当前图像攻击是否需要多层视觉 token。"""

        return self.objective_config.attack_objective in {
            "original_plus_hyperbolic",
            "hyperbolic_main",
        }

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img,all_txt_supervisions):
        device = adv_imgs_embeds.device

        U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32))
        # projection_matrix = U[:, :30] @ U[:, :30].t()
        projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t()
        # projection_matrix = U @ U.t()
        # print("txts_embeds.shape",txts_embeds.shape)
        # print("len(txt2img)",len(txt2img))
        # print("projection_matrix.shape",projection_matrix.shape)
        # print(a)

        adv_imgs_embeds = adv_imgs_embeds @ projection_matrix
        txts_embeds = txts_embeds @ projection_matrix

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        # print(it_labels)
        # print(a)
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos

        return loss
    
    def loss_func_old(self, adv_imgs_embeds, txts_embeds, txt2img, all_txt_supervisions):
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1
        
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss

    def _compute_scaled_original_loss(
        self,
        image_embeds,
        txt_embeds,
        txt2img,
        all_txt_supervisions,
        batch_size,
        scales_num,
    ):
        """按 scale chunk 计算原始 SA-AET 图像侧语义子空间 loss。"""

        loss = image_embeds.new_tensor(0.0)
        for scale_idx in range(scales_num):
            start = scale_idx * batch_size
            end = start + batch_size
            loss = loss + self.loss_func(
                image_embeds[start:end],
                txt_embeds,
                txt2img,
                all_txt_supervisions,
            )
        return loss / scales_num

    def _compute_hyperbolic_guidance_loss(
        self,
        multi_layer_tokens,
        txt_embeds,
        txt2img,
        batch_size,
        scales_num,
        apply_lambda=True,
    ):
        """计算庞加莱球双曲 ranking 攻击目标并按配置权重缩放。"""

        if self.objective_config.lambda_hyp == 0 or multi_layer_tokens is None:
            return txt_embeds.new_tensor(0.0)

        loss = txt_embeds.new_tensor(0.0)
        for scale_idx in range(scales_num):
            start = scale_idx * batch_size
            end = start + batch_size
            scale_tokens = {
                layer_name: tokens[start:end]
                for layer_name, tokens in multi_layer_tokens.items()
            }
            hyp_loss, outputs, stats = self.hyperbolic_loss(
                scale_tokens,
                txt_embeds,
                txt2img,
            )
            _ = outputs
            self.latest_hyperbolic_stats = stats
            loss = loss + hyp_loss
        mean_loss = loss / scales_num
        if apply_lambda:
            return self.objective_config.lambda_hyp * mean_loss
        return mean_loss

    def _compute_image_loss_components(
        self,
        model,
        imgs_for_loss,
        txt2img,
        all_txt_supervisions,
        txt_embeds,
        vision_extractor,
        batch_size,
        scales_num,
    ):
        """分别计算原始 SA-AET loss 与未加权双曲 loss。"""

        if self.normalization is not None:
            normalized_imgs = self.normalization(imgs_for_loss)
        else:
            normalized_imgs = imgs_for_loss

        objective = self.objective_config.attack_objective
        zero_loss = imgs_for_loss.sum() * 0.0
        original_loss = zero_loss
        raw_hyperbolic_loss = zero_loss

        if objective in {"baseline", "original_plus_hyperbolic"}:
            image_output = model.inference_image(normalized_imgs)
            image_embeds = image_output["image_feat"]
            original_loss = self._compute_scaled_original_loss(
                image_embeds,
                txt_embeds,
                txt2img,
                all_txt_supervisions,
                batch_size,
                scales_num,
            )

        if objective in {"original_plus_hyperbolic", "hyperbolic_main"}:
            if self.uses_multilevel_features and vision_extractor is not None:
                multi_layer_tokens = vision_extractor(normalized_imgs)
            elif self.uses_multilevel_features:
                raise ValueError(
                    f"attack_objective={objective} requires a HierarchicalVisionExtractor."
                )
            else:
                multi_layer_tokens = None
            raw_hyperbolic_loss = self._compute_hyperbolic_guidance_loss(
                multi_layer_tokens,
                txt_embeds,
                txt2img,
                batch_size,
                scales_num,
                apply_lambda=False,
            )

        return objective, original_loss, raw_hyperbolic_loss

    def _compose_total_image_loss(self, objective, original_loss, raw_hyperbolic_loss):
        """按配置组合 loss，并返回已加权的双曲项用于记录。"""

        lambda_hyp = float(self.objective_config.lambda_hyp)
        weighted_hyperbolic_loss = lambda_hyp * raw_hyperbolic_loss

        if objective == "baseline":
            weighted_hyperbolic_loss = raw_hyperbolic_loss * 0.0
            return original_loss, weighted_hyperbolic_loss
        if objective == "hyperbolic_main":
            return weighted_hyperbolic_loss, weighted_hyperbolic_loss

        fusion_strategy = self.hyperbolic_loss.config.gradient_fusion_strategy
        if fusion_strategy == "normalized_additive" and lambda_hyp > 0:
            original_scale = original_loss.detach().abs().clamp_min(1e-6)
            hyperbolic_scale = raw_hyperbolic_loss.detach().abs().clamp_min(1e-6)
            balanced_lambda = lambda_hyp * (original_scale / hyperbolic_scale)
            weighted_hyperbolic_loss = balanced_lambda * raw_hyperbolic_loss

        return original_loss + weighted_hyperbolic_loss, weighted_hyperbolic_loss

    def _normalize_gradient(self, grad):
        """按样本归一化梯度，避免某一项只因尺度大而主导融合。"""

        return grad / grad.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

    def _compute_attack_gradient(
        self,
        model,
        imgs_for_loss,
        gradient_target,
        txt2img,
        all_txt_supervisions,
        txt_embeds,
        vision_extractor,
        batch_size,
        scales_num,
        scales=None,
        device="cuda",
    ):
        """计算像素更新梯度。

        `candidate_guided` 保持原始 SA-AET 梯度更新，并让双曲 loss 参与
        evolution candidate 选择；`normalized_additive` 与 `pcgrad_norm` 都
        顺序计算两个分支的梯度以降低显存占用，后者会额外投影掉双曲
        梯度中的冲突分量。
        """

        objective = self.objective_config.attack_objective
        lambda_hyp = float(self.objective_config.lambda_hyp)
        fusion_strategy = self.hyperbolic_loss.config.gradient_fusion_strategy
        use_candidate_guided = (
            objective == "original_plus_hyperbolic"
            and lambda_hyp > 0
            and fusion_strategy == "candidate_guided"
        )
        use_sequential_fusion = (
            objective == "original_plus_hyperbolic"
            and lambda_hyp > 0
            and fusion_strategy in {"normalized_additive", "pcgrad_norm"}
        )

        if use_candidate_guided:
            original_imgs_for_loss = (
                self.get_scaled_imgs(gradient_target, scales, device)
                if scales is not None and scales_num > 1
                else imgs_for_loss
            )
            if self.normalization is not None:
                normalized_imgs = self.normalization(original_imgs_for_loss)
            else:
                normalized_imgs = original_imgs_for_loss
            image_output = model.inference_image(normalized_imgs)
            image_embeds = image_output["image_feat"]
            original_loss = self._compute_scaled_original_loss(
                image_embeds,
                txt_embeds,
                txt2img,
                all_txt_supervisions,
                batch_size,
                scales_num,
            )
            grad = torch.autograd.grad(
                original_loss,
                gradient_target,
                retain_graph=False,
                allow_unused=False,
            )[0]
            del image_output, image_embeds, normalized_imgs, original_imgs_for_loss

            with torch.no_grad():
                hyperbolic_imgs_for_loss = (
                    self.get_scaled_imgs(gradient_target, scales, device)
                    if scales is not None and scales_num > 1
                    else imgs_for_loss
                )
                if self.normalization is not None:
                    normalized_imgs = self.normalization(hyperbolic_imgs_for_loss)
                else:
                    normalized_imgs = hyperbolic_imgs_for_loss
                if vision_extractor is None:
                    raise ValueError(
                        f"attack_objective={objective} requires a HierarchicalVisionExtractor."
                    )
                multi_layer_tokens = vision_extractor(normalized_imgs)
                raw_hyperbolic_loss = self._compute_hyperbolic_guidance_loss(
                    multi_layer_tokens,
                    txt_embeds,
                    txt2img,
                    batch_size,
                    scales_num,
                    apply_lambda=False,
                )
                weighted_hyperbolic_loss = lambda_hyp * raw_hyperbolic_loss
                total_loss = original_loss.detach() + weighted_hyperbolic_loss.detach()
                del multi_layer_tokens, normalized_imgs, hyperbolic_imgs_for_loss

            self._record_image_loss_stats(
                objective,
                original_loss,
                weighted_hyperbolic_loss,
                total_loss,
            )
            self.latest_image_loss_stats.update(
                {
                    "gradient_fusion_strategy": fusion_strategy,
                    "hyperbolic_candidate_guided": True,
                }
            )
            return total_loss, grad

        if not use_sequential_fusion:
            loss_imgs_for_loss = (
                self.get_scaled_imgs(gradient_target, scales, device)
                if scales is not None and scales_num > 1
                else imgs_for_loss
            )
            objective, original_loss, raw_hyperbolic_loss = self._compute_image_loss_components(
                model,
                loss_imgs_for_loss,
                txt2img,
                all_txt_supervisions,
                txt_embeds,
                vision_extractor,
                batch_size,
                scales_num,
            )
            total_loss, weighted_hyperbolic_loss = self._compose_total_image_loss(
                objective,
                original_loss,
                raw_hyperbolic_loss,
            )
            grad = torch.autograd.grad(
                total_loss,
                gradient_target,
                retain_graph=False,
                allow_unused=False,
            )[0]
            self._record_image_loss_stats(
                objective,
                original_loss,
                weighted_hyperbolic_loss,
                total_loss,
            )
            return total_loss, grad

        original_imgs_for_loss = (
            self.get_scaled_imgs(gradient_target, scales, device)
            if scales is not None and scales_num > 1
            else imgs_for_loss
        )
        if self.normalization is not None:
            normalized_imgs = self.normalization(original_imgs_for_loss)
        else:
            normalized_imgs = original_imgs_for_loss
        image_output = model.inference_image(normalized_imgs)
        image_embeds = image_output["image_feat"]
        original_loss = self._compute_scaled_original_loss(
            image_embeds,
            txt_embeds,
            txt2img,
            all_txt_supervisions,
            batch_size,
            scales_num,
        )
        original_grad = torch.autograd.grad(
            original_loss,
            gradient_target,
            retain_graph=False,
            allow_unused=False,
        )[0]

        del image_output, image_embeds, normalized_imgs, original_imgs_for_loss

        hyperbolic_imgs_for_loss = (
            self.get_scaled_imgs(gradient_target, scales, device)
            if scales is not None and scales_num > 1
            else imgs_for_loss
        )
        if self.normalization is not None:
            normalized_imgs = self.normalization(hyperbolic_imgs_for_loss)
        else:
            normalized_imgs = hyperbolic_imgs_for_loss
        if vision_extractor is None:
            raise ValueError(
                f"attack_objective={objective} requires a HierarchicalVisionExtractor."
            )
        multi_layer_tokens = vision_extractor(normalized_imgs)
        raw_hyperbolic_loss = self._compute_hyperbolic_guidance_loss(
            multi_layer_tokens,
            txt_embeds,
            txt2img,
            batch_size,
            scales_num,
            apply_lambda=False,
        )
        hyperbolic_grad = torch.autograd.grad(
            raw_hyperbolic_loss,
            gradient_target,
            retain_graph=False,
            allow_unused=False,
        )[0]
        del multi_layer_tokens, normalized_imgs, hyperbolic_imgs_for_loss

        original_grad = self._normalize_gradient(original_grad)
        hyperbolic_grad = self._normalize_gradient(hyperbolic_grad)
        flat_original = original_grad.flatten(start_dim=1)
        flat_hyperbolic = hyperbolic_grad.flatten(start_dim=1)
        dot = (flat_hyperbolic * flat_original).sum(dim=1, keepdim=True)
        original_norm_sq = flat_original.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
        projection = (dot / original_norm_sq).view(-1, 1, 1, 1) * original_grad
        conflict = (dot < 0).view(-1, 1, 1, 1)
        if fusion_strategy == "pcgrad_norm":
            aligned_hyperbolic_grad = torch.where(conflict, hyperbolic_grad - projection, hyperbolic_grad)
        else:
            aligned_hyperbolic_grad = hyperbolic_grad
        grad = original_grad + lambda_hyp * aligned_hyperbolic_grad
        weighted_hyperbolic_loss = lambda_hyp * raw_hyperbolic_loss
        total_loss = original_loss.detach() + weighted_hyperbolic_loss.detach()

        self._record_image_loss_stats(
            objective,
            original_loss,
            weighted_hyperbolic_loss,
            total_loss,
        )
        self.latest_image_loss_stats.update(
            {
                "gradient_fusion_strategy": fusion_strategy,
                "gradient_conflict_ratio": float(conflict.float().mean().detach().cpu()),
                "gradient_cosine_mean": float(
                    torch.nn.functional.cosine_similarity(
                        flat_original,
                        flat_hyperbolic,
                        dim=1,
                    ).mean().detach().cpu()
                ),
            }
        )
        return total_loss, grad

    def _record_image_loss_stats(
        self,
        objective,
        original_loss,
        hyperbolic_loss,
        total_loss,
    ):
        """记录当前图像侧 loss 组合，便于实验日志追踪。"""

        def to_float(value):
            return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)

        self.latest_image_loss_stats = {
            "attack_objective": objective,
            "L_original": to_float(original_loss),
            "L_hyp_ball_rank_weighted": to_float(hyperbolic_loss),
            "L_total": to_float(total_loss),
        }

    def _compute_total_image_loss(
        self,
        model,
        imgs_for_loss,
        txt2img,
        all_txt_supervisions,
        txt_embeds,
        vision_extractor,
        batch_size,
        scales_num,
    ):
        """按 `attack_objective` 计算图像侧攻击 loss。"""

        objective, original_loss, raw_hyperbolic_loss = self._compute_image_loss_components(
            model,
            imgs_for_loss,
            txt2img,
            all_txt_supervisions,
            txt_embeds,
            vision_extractor,
            batch_size,
            scales_num,
        )
        total_loss, hyperbolic_loss = self._compose_total_image_loss(
            objective,
            original_loss,
            raw_hyperbolic_loss,
        )
        self._record_image_loss_stats(
            objective,
            original_loss,
            hyperbolic_loss,
            total_loss,
        )
        return total_loss

    def rand3Num(self): ### num1 -> adv num2-> clean num3->last
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2
            
            if 1 <= num3 <= 100 and num1 < num3 and num3 < num2:
                break

        return (num1, num2, num3)

    def txt_guided_attack(self, model, imgs, txt2img, all_txt_supervisions,device, vision_extractor, scales=None, txt_embeds=None):

        model.eval()

        b, _, _, _ = imgs.shape

        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        last_adv_imgs = None

        start_time = time.time()
        ratio_list = []

        for step in range(self.steps):  # self.steps=10
            if last_adv_imgs != None:
                samples = []
                clone_adv_imgs = adv_imgs.clone()
                loss_list = []
                for k in range(self.sample_numbers):
                    samples.append(self.rand3Num())
                for sample in samples:
                    adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                sample[2] / 100) * last_adv_imgs
                    adv_imgs = adv_imgs.detach().requires_grad_(True)

                    model.zero_grad()
                    with torch.enable_grad():
                        loss, grad = self._compute_attack_gradient(
                            model,
                            adv_imgs,
                            adv_imgs,
                            txt2img,
                            all_txt_supervisions,
                            txt_embeds,
                            vision_extractor,
                            b,
                            1,
                            device=device,
                        )
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                    perturbation = self.step_size * grad.sign()

                    adv_imgs = clone_adv_imgs.detach() + perturbation
                    adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)



                    model.zero_grad()
                    with torch.no_grad():
                        loss = self._compute_total_image_loss(
                            model,
                            adv_imgs,
                            txt2img,
                            all_txt_supervisions,
                            txt_embeds,
                            vision_extractor,
                            b,
                            1,
                        )
                    loss_list.append(loss.item())
                #candidate_index = loss_list.index(max(loss_list))

                candidate_index = loss_list.index(max(loss_list))
                ratio_list.append(samples[candidate_index])

                adv_imgs = (samples[candidate_index][0] / 100) * clone_adv_imgs + (
                            samples[candidate_index][1] / 100) * imgs + (
                                       samples[candidate_index][2] / 100) * last_adv_imgs
                adv_imgs = adv_imgs.detach().requires_grad_(True)

                model.zero_grad()
                with torch.enable_grad():
                    loss, grad = self._compute_attack_gradient(
                        model,
                        adv_imgs,
                        adv_imgs,
                        txt2img,
                        all_txt_supervisions,
                        txt_embeds,
                        vision_extractor,
                        b,
                        scales_num,
                        scales=scales,
                        device=device,
                    )
                print("loss", loss)

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                perturbation = self.step_size * grad.sign()

                adv_imgs = clone_adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                last_adv_imgs = clone_adv_imgs.clone()
            else:
                last_adv_imgs = adv_imgs.clone()
                adv_imgs = adv_imgs.detach().requires_grad_(True)

                model.zero_grad()
                with torch.enable_grad():
                    loss, grad = self._compute_attack_gradient(
                        model,
                        adv_imgs,
                        adv_imgs,
                        txt2img,
                        all_txt_supervisions,
                        txt_embeds,
                        vision_extractor,
                        b,
                        scales_num,
                        scales=scales,
                        device=device,
                    )
                print("loss",loss)
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                perturbation = self.step_size * grad.sign()
                adv_imgs = adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"The function execution time: {elapsed_time} seconds")

        return adv_imgs, last_adv_imgs

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])

        reverse_transform = transforms.Resize(ori_shape,
                                              interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]),
                           int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)

            result.append(reversed_imgs)

        return torch.cat([imgs, ] + result, 0)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, last_adv_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                if adv_img_embeds == None:
                    loss = self.loss_func(replace_embeds, img_embeds, i)
                else:
                    loss = self.text_ratios[0] * self.loss_func(replace_embeds, img_embeds, i) + self.text_ratios[
                        1] * self.loss_func(replace_embeds, adv_img_embeds, i) + self.text_ratios[2] * self.loss_func(
                        replace_embeds, last_adv_img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
