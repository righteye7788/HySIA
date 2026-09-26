"""在 MSCOCO 上离线训练 LearnableHyperbolicAdapter。

源 VLP encoder 全程冻结。训练使用每张图的全部正 caption；checkpoint 仅由
image-disjoint COCO validation ranking loss 选择。训练完成后，脚本在 COCO val
上执行独立的 semantic-granularity radius probe，该 probe 不参与梯度或模型选择。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import BertTokenizer

from dataset import pre_caption
from hierarchical_visual_attack import (
    HierarchicalVisionExtractor,
    HyperbolicGuidanceConfig,
    HyperbolicGuidanceLoss,
    LearnableHyperbolicAdapter,
    LearnableHyperbolicAdapterConfig,
    save_adapter_checkpoint,
)
from models import clip
from models.model_retrieval import ALBEF


yaml = YAML(typ="safe")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
UNIT_LEVELS = {"global": 1, "relation": 2, "object": 3, "attribute": 4}
UNIT_ORDER = tuple(UNIT_LEVELS)
ALBEF_ALLOWED_MISSING_PREFIXES = (
    "visual_encoder_m.",
    "vision_proj_m.",
    "text_encoder_m.",
    "text_proj_m.",
    "image_queue",
    "text_queue",
    "idx_queue",
    "queue_ptr",
)


class CocoGroupedCaptionDataset(Dataset):
    """Group Karpathy annotations by image and return every positive caption."""

    def __init__(self, ann_file: str, transform, image_root: str, max_words: int = 30):
        with open(ann_file, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        grouped: "OrderedDict[str, Dict]" = OrderedDict()
        for annotation in annotations:
            image_path = annotation["image"]
            record = grouped.setdefault(
                image_path,
                {
                    "image": image_path,
                    "image_id": annotation.get("image_id", len(grouped)),
                    "captions": [],
                },
            )
            captions = annotation["caption"]
            if isinstance(captions, str):
                captions = [captions]
            record["captions"].extend(pre_caption(text, max_words) for text in captions)

        self.records = list(grouped.values())
        self.transform = transform
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = os.path.join(self.image_root, record["image"])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), tuple(record["captions"]), record["image_id"], record["image"]

    @staticmethod
    def collate_fn(batch):
        images, caption_groups, image_ids, image_paths = zip(*batch)
        return torch.stack(images, dim=0), caption_groups, list(image_ids), image_paths


class SemanticUnitExtractor:
    """Extract mutually exclusive global/relation/object/attribute units."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy is required for semantic granularity validation.") from exc
        try:
            self.nlp = spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError(f"Unable to load spaCy model: {model_name}") from exc

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def extract(self, caption: str) -> Dict[str, List[str]]:
        caption = self._normalize(caption)
        doc = self.nlp(caption)
        attributes, objects, relations = [], [], []
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                object_text = self._normalize(token.lemma_ or token.text)
                if object_text:
                    objects.append(object_text)
                modifiers = [
                    child
                    for child in token.children
                    if child.dep_ in {"amod", "compound", "nummod"} or child.pos_ == "ADJ"
                ]
                if modifiers:
                    phrase_tokens = sorted(modifiers + [token], key=lambda item: item.i)
                    phrase = self._normalize(" ".join(item.text for item in phrase_tokens))
                    if phrase and phrase != object_text:
                        attributes.append(phrase)
            if token.pos_ == "VERB":
                arguments = [
                    child
                    for child in token.children
                    if child.dep_ in {
                        "nsubj",
                        "nsubjpass",
                        "dobj",
                        "obj",
                        "iobj",
                        "pobj",
                        "attr",
                        "dative",
                        "oprd",
                    }
                    or child.pos_ in {"NOUN", "PROPN", "PRON"}
                ]
                relation_tokens = sorted(arguments + [token], key=lambda item: item.i)
                phrase = self._normalize(
                    " ".join(
                        (item.lemma_ or item.text) if item == token else item.text
                        for item in relation_tokens
                    )
                )
                if phrase and len(relation_tokens) >= 2:
                    relations.append(phrase)

        def unique(values: Iterable[str]) -> List[str]:
            return list(OrderedDict((value, None) for value in values if value))

        # Exact cross-category duplicates are assigned to the more specific
        # category: attribute > object > relation > global.
        attributes = unique(attributes)
        occupied = set(attributes)
        objects = [value for value in unique(objects) if value not in occupied]
        occupied.update(objects)
        relations = [value for value in unique(relations) if value not in occupied]
        occupied.update(relations)
        global_units = [caption] if caption and caption not in occupied else []
        return {
            "global": global_units,
            "relation": relations,
            "object": objects,
            "attribute": attributes,
        }


def parse_hier_feature_layers(layer_arg, source_model="CLIP_ViT"):
    feature_layers = []
    for item in str(layer_arg).split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            prefix = "blocks" if str(source_model) in {"ALBEF", "TCL"} else "transformer.resblocks"
            item = f"{prefix}.{item}"
        feature_layers.append(item)
    return feature_layers


def resolve_visual_sequence_first(source_model):
    return str(source_model) not in {"ALBEF", "TCL"}


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    required = ["train_file", "val_file", "image_root"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError("Missing required adapter training config: " + ", ".join(missing))
    for key in ("train_file", "val_file"):
        if not os.path.isfile(config[key]):
            raise FileNotFoundError(f"{key} does not exist: {config[key]}")
    if not os.path.isdir(config["image_root"]):
        raise FileNotFoundError(f"image_root does not exist: {config['image_root']}")
    if config.get("adapter_train_data", "mscoco_train") != "mscoco_train":
        raise ValueError("This entrypoint requires adapter_train_data=mscoco_train.")
    source_model = config.get("source_model", "CLIP_ViT")
    if source_model not in {"CLIP_ViT", "CLIP_CNN", "ALBEF"}:
        raise ValueError(f"Unsupported source_model: {source_model}")
    if source_model == "ALBEF" and not os.path.isfile(config.get("albef_ckpt", "")):
        raise FileNotFoundError(f"albef_ckpt does not exist: {config.get('albef_ckpt')}")


def resolve_image_size(config, model):
    if hasattr(model, "visual") and hasattr(model.visual, "input_resolution"):
        return int(model.visual.input_resolution)
    return int(config.get("image_res", 384))


def build_image_transform(source_model: str, config: Dict, model):
    image_size = resolve_image_size(config, model)
    if source_model == "ALBEF":
        return transforms.Compose(
            [transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC), transforms.ToTensor()]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def flatten_caption_groups(caption_groups: Sequence[Sequence[str]]) -> Tuple[List[str], List[int]]:
    texts, txt2img = [], []
    for image_index, captions in enumerate(caption_groups):
        for caption in captions:
            texts.append(caption)
            txt2img.append(image_index)
    return texts, txt2img


def load_clip_vit(config, device):
    tokenizer = BertTokenizer.from_pretrained(config["source_text_encoder"])
    model_path = config.get("clip_model_path") or "/home/data/share_data/multimodal/models/clip/ViT-B-16.pt"
    model, _ = clip.load(model_path, device=device)
    model.set_tokenizer(tokenizer)
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.adapter_load_report = {"missing_keys": [], "unexpected_keys": []}
    return model, tokenizer


def load_albef(config, device):
    tokenizer = BertTokenizer.from_pretrained(config["source_text_encoder"])
    model = ALBEF(config=config, text_encoder=config["source_text_encoder"], tokenizer=tokenizer)
    checkpoint = torch.load(config["albef_ckpt"], map_location="cpu")
    state_dict = dict(checkpoint.get("model", checkpoint))
    for key in list(state_dict):
        if "bert" in key:
            state_dict[key.replace("bert.", "")] = state_dict.pop(key)
    incompatible = model.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key for key in incompatible.missing_keys if not key.startswith(ALBEF_ALLOWED_MISSING_PREFIXES)
    ]
    if critical_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "ALBEF checkpoint is incompatible with the primary encoder: "
            f"critical_missing={critical_missing}, unexpected={incompatible.unexpected_keys}"
        )
    model.adapter_load_report = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "critical_missing_keys": critical_missing,
    }
    model.to(device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def load_source_model(config, device):
    if config.get("source_model", "CLIP_ViT") == "ALBEF":
        return load_albef(config, device)
    return load_clip_vit(config, device)


def build_adapter(config, visual_tokens_by_layer, text_embeds):
    adapter_config = LearnableHyperbolicAdapterConfig(
        layer_dims={name: int(tokens.shape[-1]) for name, tokens in visual_tokens_by_layer.items()},
        text_dim=int(text_embeds.shape[-1]),
        adapter_dim=int(config.get("adapter_dim", 256)),
        hidden_dim=config.get("adapter_hidden_dim"),
        radius_min=float(config.get("radius_min", 0.1)),
        radius_max=float(config.get("radius_max", 0.8)),
        input_normalization=config.get("adapter_input_normalization", "layernorm_l2"),
        curvature=float(config.get("hyperbolic_curvature", 1.0)),
        eps=float(config.get("hyperbolic_eps", 1e-5)),
    )
    return LearnableHyperbolicAdapter(adapter_config)


def build_guidance_loss(config, adapter):
    guidance_config = HyperbolicGuidanceConfig(
        adapter_dim=int(config.get("adapter_dim", 256)),
        adapter_input_normalization=config.get("adapter_input_normalization", "layernorm_l2"),
        adapter_train_data=config.get("adapter_train_data", "mscoco_train"),
        adapter_train_objective=config.get("adapter_train_objective", "clean_ranking_logmeanexp"),
        radius_min=float(config.get("radius_min", 0.1)),
        radius_max=float(config.get("radius_max", 0.8)),
        curvature=float(config.get("hyperbolic_curvature", 1.0)),
        eps=float(config.get("hyperbolic_eps", 1e-5)),
        alignment_temperature=float(config.get("alignment_temperature", config.get("hyperbolic_tau", 0.1))),
        radius_temperature=float(config.get("radius_temperature", config.get("hyperbolic_tau", 0.1))),
        layer_weight_strategy=config.get("layer_weight_strategy", "fixed_depth"),
        hard_negative_topk=int(config.get("hard_negative_topk", 5)),
        ranking_temperature=float(config.get("ranking_temperature", 0.1)),
    )
    return HyperbolicGuidanceLoss(guidance_config, adapter=adapter)


def encode_frozen_features(model, tokenizer, extractor, image_normalize, images, texts, max_length):
    with torch.no_grad():
        text_input = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(images.device)
        text_embeds = model.inference_text(text_input)["text_feat"]
        visual_tokens = extractor(image_normalize(images))
    return visual_tokens, text_embeds


def compute_radius_regularization(visual_radius_by_layer, text_radius, config):
    radius_min = float(config.get("radius_min", 0.1))
    radius_max = float(config.get("radius_max", 0.8))
    visual_ratio = float(config.get("visual_radius_upper_target_ratio", 0.85))
    text_ratio = float(config.get("text_radius_upper_target_ratio", 0.80))
    min_std = float(config.get("radius_variance_min_std", 0.02))
    visual_target = radius_min + visual_ratio * (radius_max - radius_min)
    text_target = radius_min + text_ratio * (radius_max - radius_min)
    visual_radii = torch.cat([radius.reshape(-1) for radius in visual_radius_by_layer.values()])
    text_radii = text_radius.reshape(-1)

    visual_upper = (
        torch.relu(visual_radii - visual_target) / max(radius_max - visual_target, 1e-12)
    ).pow(2).mean()
    text_upper = (
        torch.relu(text_radii - text_target) / max(radius_max - text_target, 1e-12)
    ).pow(2).mean()
    visual_std = visual_radii.std(unbiased=False)
    text_std = text_radii.std(unbiased=False)
    visual_variance = (torch.relu(visual_radii.new_tensor(min_std) - visual_std) / min_std).pow(2)
    text_variance = (torch.relu(text_radii.new_tensor(min_std) - text_std) / min_std).pow(2)
    losses = {
        "visual_upper": visual_upper,
        "text_upper": text_upper,
        "visual_variance": visual_variance,
        "text_variance": text_variance,
    }
    with torch.no_grad():
        stats = {
            "visual_raw_radius_mean": float(visual_radii.mean().cpu()),
            "visual_raw_radius_std": float(visual_std.cpu()),
            "visual_raw_radius_min": float(visual_radii.min().cpu()),
            "visual_raw_radius_max": float(visual_radii.max().cpu()),
            "text_raw_radius_mean": float(text_radii.mean().cpu()),
            "text_raw_radius_std": float(text_std.cpu()),
            "text_raw_radius_min": float(text_radii.min().cpu()),
            "text_raw_radius_max": float(text_radii.max().cpu()),
            "L_visual_radius_upper": float(visual_upper.cpu()),
            "L_text_radius_upper": float(text_upper.cpu()),
            "L_visual_radius_variance": float(visual_variance.cpu()),
            "L_text_radius_variance": float(text_variance.cpu()),
        }
    return losses, stats


def aggregate_relation_distances(loss_fn, relation_matrix):
    return loss_fn.aggregate_layer_distances(relation_matrix)


def pairwise_accuracy(loss_fn, relation_matrix, txt2img):
    distances = aggregate_relation_distances(loss_fn, relation_matrix)
    positive_mask, negative_mask = loss_fn.build_caption_masks(
        txt2img, distances.shape[0], distances.device, distances.dtype
    )
    correct, total, r1_correct, valid_r1 = 0.0, 0, 0, 0
    for image_index in range(distances.shape[0]):
        positives = distances[image_index][positive_mask[image_index].bool()]
        negatives = distances[image_index][negative_mask[image_index].bool()]
        if positives.numel() and negatives.numel():
            correct += float((positives[:, None] < negatives[None, :]).float().sum().cpu())
            total += positives.numel() * negatives.numel()
            best_caption = int(distances[image_index].argmin().item())
            r1_correct += int(bool(positive_mask[image_index, best_caption].item()))
            valid_r1 += 1
    return correct / max(total, 1), r1_correct / max(valid_r1, 1)


def evaluate_adapter(adapter, loss_fn, model, tokenizer, extractor, loader, image_normalize, device, max_length):
    adapter.eval()
    totals = defaultdict(float)
    total_images = 0
    with torch.no_grad():
        for images, caption_groups, _image_ids, _paths in loader:
            images = images.to(device)
            texts, txt2img = flatten_caption_groups(caption_groups)
            visual_tokens, text_embeds = encode_frozen_features(
                model, tokenizer, extractor, image_normalize, images, texts, max_length
            )
            visual_tangent, text_tangent, visual_h, text_h, _, _ = adapter(visual_tokens, text_embeds)
            relation = loss_fn.compute_relation_matrix(visual_tangent, text_tangent, visual_h, text_h)
            loss = loss_fn.compute_ranking_loss(relation, txt2img)
            stats = loss_fn.compute_stats(relation, txt2img)
            accuracy, r1 = pairwise_accuracy(loss_fn, relation, txt2img)
            batch_images = images.shape[0]
            totals["val_ranking_loss"] += float(loss.cpu()) * batch_images
            totals["val_pairwise_accuracy"] += accuracy * batch_images
            totals["val_batch_i2t_r1"] += r1 * batch_images
            for key in ("positive_caption_distance_mean", "negative_caption_distance_mean"):
                totals[f"val_{key}"] += float(stats[key]) * batch_images
            total_images += batch_images
    result = {key: value / max(total_images, 1) for key, value in totals.items()}
    result["val_images"] = total_images
    return result


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    x_rank, y_rank = rankdata(x), rankdata(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def bootstrap_spearman(records, radius_field, iterations, seed):
    by_image = defaultdict(list)
    for record in records:
        by_image[str(record["image_id"])].append(record)
    image_ids = list(by_image)
    if len(image_ids) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        rows = [row for image_id in sampled for row in by_image[str(image_id)]]
        rho = spearman_rho(
            [row["granularity_level"] for row in rows], [row[radius_field] for row in rows]
        )
        if math.isfinite(rho):
            values.append(rho)
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def write_semantic_plot(summary, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skip semantic granularity plot.")
        return None
    levels = list(UNIT_ORDER)
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for field, label, color, offset in (
        ("aligned_visual_radius", "Aligned visual radius", "#32688E", -0.08),
        ("text_radius", "Text radius", "#D17A22", 0.08),
    ):
        means = [summary["level_stats"][level][field]["mean"] for level in levels]
        errors = [
            0.5
            * (
                summary["level_stats"][level][field]["ci95_high"]
                - summary["level_stats"][level][field]["ci95_low"]
            )
            for level in levels
        ]
        ax.errorbar(x + offset, means, yerr=errors, marker="o", capsize=4, label=label, color=color)
    ax.set_xticks(x, [level.title() for level in levels])
    ax.set_ylabel(r"Mean hyperbolic radius $d_c(0, \psi)$")
    ax.set_title("MSCOCO semantic granularity probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def summarize_semantic_records(records, adapter, config, seed):
    radius_fields = ("aligned_visual_radius", "text_radius")
    level_stats = {}
    for level in UNIT_ORDER:
        rows = [row for row in records if row["unit_type"] == level]
        level_stats[level] = {}
        for field in radius_fields:
            values = np.asarray([row[field] for row in rows], dtype=float)
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std()) if len(values) else float("nan")
            half_width = 1.96 * std / math.sqrt(len(values)) if len(values) else float("nan")
            level_stats[level][field] = {
                "count": int(len(values)),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }

    rhos = {
        field: spearman_rho(
            [row["granularity_level"] for row in records], [row[field] for row in records]
        )
        for field in radius_fields
    }
    rho_cis = {
        field: bootstrap_spearman(
            records,
            field,
            int(config.get("semantic_bootstrap_iters", 1000)),
            seed,
        )
        for field in radius_fields
    }
    by_caption = defaultdict(dict)
    for row in records:
        by_caption[(str(row["image_id"]), int(row["caption_index"]))][row["unit_type"]] = {
            field: row[field] for field in radius_fields
        }
    complete = [values for values in by_caption.values() if all(level in values for level in UNIT_ORDER)]
    ordering_accuracy = {}
    means_monotonic = {}
    for field in radius_fields:
        ordered = sum(
            int(
                all(
                    values[UNIT_ORDER[index]][field] < values[UNIT_ORDER[index + 1]][field]
                    for index in range(3)
                )
            )
            for values in complete
        )
        ordering_accuracy[field] = ordered / len(complete) if complete else 0.0
        level_means = [level_stats[level][field]["mean"] for level in UNIT_ORDER]
        means_monotonic[field] = all(
            math.isfinite(level_means[index])
            and math.isfinite(level_means[index + 1])
            and level_means[index] < level_means[index + 1]
            for index in range(3)
        )
    # With this project's exp_0 convention d_c(0, exp_0(v)) = 2 ||v||.
    lower = 2.0 * float(adapter.radius_min) + 1e-3
    upper = 2.0 * float(adapter.radius_max) - 1e-3
    visual_values = np.asarray([row["aligned_visual_radius"] for row in records])
    text_values = np.asarray([row["text_radius"] for row in records])
    saturation = {
        "visual_lower": float(np.mean(visual_values <= lower)) if len(visual_values) else 0.0,
        "visual_upper": float(np.mean(visual_values >= upper)) if len(visual_values) else 0.0,
        "text_lower": float(np.mean(text_values <= lower)) if len(text_values) else 0.0,
        "text_upper": float(np.mean(text_values >= upper)) if len(text_values) else 0.0,
    }
    max_saturation = max(saturation.values())
    positive_rho = all(
        math.isfinite(rhos[field]) and rhos[field] > 0 and rho_cis[field][0] > 0
        for field in radius_fields
    )
    increasing_means = all(means_monotonic.values())
    ordering_above_chance = all(value > 0.5 for value in ordering_accuracy.values())
    supported = bool(
        positive_rho
        and increasing_means
        and ordering_above_chance
        and max_saturation < 0.05
    )
    return {
        "unit_level_definition": UNIT_LEVELS,
        "level_stats": level_stats,
        "aligned_visual_radius_spearman": rhos["aligned_visual_radius"],
        "aligned_visual_radius_spearman_bootstrap_ci95": rho_cis["aligned_visual_radius"],
        "text_radius_spearman": rhos["text_radius"],
        "text_radius_spearman_bootstrap_ci95": rho_cis["text_radius"],
        "complete_chain_captions": len(complete),
        "complete_chain_ordering_accuracy": ordering_accuracy,
        "level_means_strictly_increasing": means_monotonic,
        "hyperbolic_radius_bounds": {"lower": lower, "upper": upper},
        "radius_saturation": saturation,
        "semantic_granularity_supported": supported,
        "support_criteria": {
            "positive_spearman_with_positive_ci_for_both_modalities": positive_rho,
            "strictly_increasing_level_means_for_both_modalities": increasing_means,
            "ordering_accuracy_above_chance_for_both_modalities": ordering_above_chance,
            "max_saturation_below_5_percent": max_saturation < 0.05,
        },
    }


def run_semantic_probe(
    adapter,
    loss_fn,
    model,
    tokenizer,
    extractor,
    dataset,
    image_normalize,
    device,
    config,
    output_dir,
    seed,
):
    if not as_bool(config.get("semantic_probe_enabled", True)):
        return {"enabled": False}
    semantic_extractor = SemanticUnitExtractor(config.get("spacy_model", "en_core_web_sm"))
    max_images = min(int(config.get("semantic_probe_max_images", 5000)), len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(max_images))),
        batch_size=int(config.get("semantic_probe_batch_size", 2)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        collate_fn=dataset.collate_fn,
    )
    text_chunk_size = int(config.get("semantic_text_batch_size", 64))
    max_length = int(config.get("max_text_length", 30))
    layer_weights = loss_fn.compute_layer_weights(
        len(adapter.layer_names), device, torch.float32
    ).detach().cpu().numpy()
    unit_records = []
    adapter.eval()
    with torch.no_grad():
        for images, caption_groups, image_ids, _paths in loader:
            images = images.to(device)
            visual_tokens = extractor(image_normalize(images))
            units = []
            for local_image_index, captions in enumerate(caption_groups):
                for caption_index, caption in enumerate(captions):
                    extracted = semantic_extractor.extract(caption)
                    for unit_type in UNIT_ORDER:
                        for text in extracted[unit_type]:
                            units.append(
                                {
                                    "local_image_index": local_image_index,
                                    "image_id": image_ids[local_image_index],
                                    "caption_index": caption_index,
                                    "caption": caption,
                                    "unit_type": unit_type,
                                    "granularity_level": UNIT_LEVELS[unit_type],
                                    "unit_text": text,
                                }
                            )
            for start in range(0, len(units), text_chunk_size):
                chunk = units[start : start + text_chunk_size]
                texts = [item["unit_text"] for item in chunk]
                text_input = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                text_embeds = model.inference_text(text_input)["text_feat"]
                _, _, visual_h, text_h, _, _ = adapter(visual_tokens, text_embeds)
                alignment, _, _, _ = loss_fn.compute_alignment_weights(visual_h, text_h)
                visual_h_radius = torch.stack(
                    [adapter.ball.norm(layer_features) for layer_features in visual_h.values()],
                    dim=1,
                )
                aligned_by_layer = (alignment * visual_h_radius.unsqueeze(-1)).sum(dim=2)
                aligned_visual = (
                    aligned_by_layer
                    * torch.as_tensor(layer_weights, device=device).view(1, -1, 1)
                ).sum(dim=1)
                text_h_radius = adapter.ball.norm(text_h)
                for chunk_index, item in enumerate(chunk):
                    local_image = item["local_image_index"]
                    unit_records.append(
                        {
                            **item,
                            "aligned_visual_radius": float(aligned_visual[local_image, chunk_index].cpu()),
                            "text_radius": float(text_h_radius[chunk_index].cpu()),
                        }
                    )

    grouped = defaultdict(list)
    for record in unit_records:
        key = (str(record["image_id"]), record["caption_index"], record["unit_type"])
        grouped[key].append(record)
    records = []
    for rows in grouped.values():
        first = rows[0]
        records.append(
            {
                "image_id": first["image_id"],
                "caption_index": first["caption_index"],
                "caption": first["caption"],
                "unit_type": first["unit_type"],
                "granularity_level": first["granularity_level"],
                "unit_count": len(rows),
                "aligned_visual_radius": float(
                    np.mean([row["aligned_visual_radius"] for row in rows])
                ),
                "text_radius": float(np.mean([row["text_radius"] for row in rows])),
            }
        )

    records_path = output_dir / "semantic_granularity_records.csv"
    fields = list(records[0]) if records else [
        "image_id",
        "caption_index",
        "caption",
        "unit_type",
        "granularity_level",
        "unit_count",
        "aligned_visual_radius",
        "text_radius",
    ]
    with open(records_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize_semantic_records(records, adapter, config, seed)
    summary.update(
        {
            "enabled": True,
            "images": max_images,
            "caption_level_records": len(records),
            "unit_records_before_caption_averaging": len(unit_records),
            "records_file": str(records_path),
        }
    )
    summary_path = output_dir / "semantic_granularity_summary.json"
    summary["plot_file"] = write_semantic_plot(
        summary, output_dir / "semantic_granularity_radius.png"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def train(args, config):
    validate_config(config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_model = config.get("source_model", "CLIP_ViT")
    model, tokenizer = load_source_model(config, device)
    feature_layers = parse_hier_feature_layers(
        config.get("hier_feature_layers", "2,5,8,11"), source_model
    )
    extractor = HierarchicalVisionExtractor(
        model,
        feature_layers,
        token_mode=config.get("token_granularity", "patch"),
        sequence_first=resolve_visual_sequence_first(source_model),
        cnn_token_grid_size=config.get("cnn_token_grid_size"),
    ).eval()
    transform = build_image_transform(source_model, config, model)
    image_normalize = transforms.Normalize(CLIP_MEAN, CLIP_STD)
    max_length = int(config.get("max_text_length", 30 if source_model == "ALBEF" else 77))

    train_dataset = CocoGroupedCaptionDataset(
        config["train_file"], transform, config["image_root"], max_length
    )
    val_dataset = CocoGroupedCaptionDataset(
        config["val_file"], transform, config["image_root"], max_length
    )
    batch_size = int(config.get("batch_size_train", 8 if source_model == "ALBEF" else 32))
    workers = int(config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=train_dataset.collate_fn,
    )
    validation_max_images = min(int(config.get("validation_max_images", 1000)), len(val_dataset))
    val_loader = DataLoader(
        Subset(val_dataset, list(range(validation_max_images))),
        batch_size=int(config.get("batch_size_val", batch_size)),
        shuffle=False,
        num_workers=workers,
        collate_fn=val_dataset.collate_fn,
    )

    epochs = int(config.get("schedular", {}).get("epochs", 5))
    max_train_batches = config.get("max_train_batches")
    train_batches_per_epoch = min(len(train_loader), int(max_train_batches)) if max_train_batches is not None else len(train_loader)
    total_steps = epochs * train_batches_per_epoch
    log_interval = int(config.get("log_interval", 20))
    early_config = config.get("early_stop", {}) or {}
    patience_epochs = int(early_config.get("patience_epochs", 2))
    min_delta = float(early_config.get("min_delta", 1e-3))
    early_stop_enabled = as_bool(early_config.get("enabled", True))

    print(
        "Adapter training data: "
        f"images={len(train_dataset)}, captions={sum(len(item['captions']) for item in train_dataset.records)}, "
        f"batch_size_images={batch_size}, validation_images={validation_max_images}, "
        f"epochs={epochs}, total_steps={total_steps}"
    )

    adapter = loss_fn = optimizer = scheduler = None
    best_state = None
    best_validation = None
    best_epoch = stale_epochs = global_step = 0
    latest_stats = {}
    validation_history = []
    start_time = time.time()

    for epoch in range(epochs):
        for batch_index, (images, caption_groups, _image_ids, _paths) in enumerate(train_loader):
            if batch_index >= train_batches_per_epoch:
                break
            images = images.to(device)
            texts, txt2img = flatten_caption_groups(caption_groups)
            visual_tokens, text_embeds = encode_frozen_features(
                model, tokenizer, extractor, image_normalize, images, texts, max_length
            )
            if adapter is None:
                adapter = build_adapter(config, visual_tokens, text_embeds).to(device)
                loss_fn = build_guidance_loss(config, adapter).to(device)
                opt_config = config.get("optimizer", {})
                optimizer = torch.optim.AdamW(
                    adapter.parameters(),
                    lr=float(opt_config.get("lr", 1e-4)),
                    weight_decay=float(opt_config.get("weight_decay", 0.01)),
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_steps),
                    eta_min=float(config.get("schedular", {}).get("min_lr", 1e-6)),
                )

            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            visual_tangent, text_tangent, visual_h, text_h, visual_radius, text_radius = adapter(
                visual_tokens, text_embeds
            )
            relation = loss_fn.compute_relation_matrix(visual_tangent, text_tangent, visual_h, text_h)
            rank_loss = loss_fn.compute_ranking_loss(relation, txt2img)
            radius_losses, radius_stats = compute_radius_regularization(visual_radius, text_radius, config)
            loss = rank_loss
            loss = loss + float(config.get("lambda_visual_radius_upper", 0.0)) * radius_losses["visual_upper"]
            loss = loss + float(config.get("lambda_text_radius_upper", 0.0)) * radius_losses["text_upper"]
            loss = loss + float(config.get("lambda_visual_radius_variance", 0.0)) * radius_losses["visual_variance"]
            loss = loss + float(config.get("lambda_text_radius_variance", 0.0)) * radius_losses["text_variance"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite adapter loss at step {global_step + 1}.")
            loss.backward()
            gradient_norm = torch.linalg.vector_norm(
                torch.stack(
                    [parameter.grad.detach().norm() for parameter in adapter.parameters() if parameter.grad is not None]
                )
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite adapter gradient at step {global_step + 1}.")
            optimizer.step()
            scheduler.step()
            global_step += 1

            ranking_stats = loss_fn.compute_stats(relation.detach(), txt2img)
            accuracy, batch_r1 = pairwise_accuracy(loss_fn, relation.detach(), txt2img)
            latest_stats = {
                **ranking_stats,
                **radius_stats,
                "epoch": epoch + 1,
                "global_step": global_step,
                "total_steps": total_steps,
                "batch_index": batch_index + 1,
                "batches_per_epoch": train_batches_per_epoch,
                "batch_size_images": images.shape[0],
                "num_texts": len(texts),
                "L_adapter_train": float(loss.detach().cpu()),
                "L_clean_ranking": float(rank_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "train_pairwise_accuracy": accuracy,
                "train_batch_i2t_r1": batch_r1,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start_time,
            }
            if batch_index == 0 or (batch_index + 1) % log_interval == 0 or batch_index + 1 == train_batches_per_epoch:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_index + 1}/{train_batches_per_epoch}] "
                    f"Loss={latest_stats['L_adapter_train']:.6f} PairAcc={accuracy:.4f} R1={batch_r1:.4f}"
                )
                print(json.dumps(latest_stats, ensure_ascii=False))

        if adapter is None:
            raise RuntimeError("No training batch was processed.")
        validation = evaluate_adapter(
            adapter, loss_fn, model, tokenizer, extractor, val_loader, image_normalize, device, max_length
        )
        validation.update({"epoch": epoch + 1, "global_step": global_step})
        validation_history.append(validation)
        print("Validation " + json.dumps(validation, ensure_ascii=False))
        current = validation["val_ranking_loss"]
        if best_validation is None or current < best_validation - min_delta:
            best_validation = current
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stop_enabled and stale_epochs >= patience_epochs:
            print(f"Early stop: val_ranking_loss did not improve for {stale_epochs} epochs.")
            break

    final_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
    if best_state is None:
        best_state = final_state
        best_validation = validation_history[-1]["val_ranking_loss"]
        best_epoch = validation_history[-1]["epoch"]
    adapter.load_state_dict(best_state)

    output_dir = Path(config.get("output_dir", "checkpoints/hyperbolic_adapter"))
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_summary = run_semantic_probe(
        adapter,
        loss_fn,
        model,
        tokenizer,
        extractor,
        val_dataset,
        image_normalize,
        device,
        config,
        output_dir,
        args.seed,
    )
    print("Semantic granularity " + json.dumps(semantic_summary, ensure_ascii=False))

    checkpoint_path = output_dir / config.get(
        "adapter_checkpoint_name", "learnable_hyperbolic_adapter_mscoco.pt"
    )
    source_checkpoint = config.get("albef_ckpt") if source_model == "ALBEF" else config.get("clip_model_path")
    image_size = resolve_image_size(config, model)
    metadata = {
        "checkpoint_schema_version": 2,
        "source_model": source_model,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": file_sha256(source_checkpoint) if source_checkpoint else None,
        "source_checkpoint_load_report": getattr(model, "adapter_load_report", {}),
        "train_file": config["train_file"],
        "val_file": config["val_file"],
        "image_root": config["image_root"],
        "preprocessing": {
            "resize": [image_size, image_size] if source_model == "ALBEF" else image_size,
            "center_crop": source_model != "ALBEF",
            "normalize_mean": CLIP_MEAN,
            "normalize_std": CLIP_STD,
        },
        "feature_layers": feature_layers,
        "alignment_normalization": "per_layer_token_softmax",
        "layer_aggregation": "normalized_layer_weights",
        "caption_sampling": "all_captions_per_image",
        "adapter_train_data": "mscoco_train",
        "adapter_train_objective": config.get("adapter_train_objective", "clean_ranking_logmeanexp"),
        "best_monitor": "val_ranking_loss",
        "best_monitor_value": best_validation,
        "best_epoch": best_epoch,
        "last_step": global_step,
        "latest_stats": latest_stats,
        "validation_history": validation_history,
        "semantic_granularity": semantic_summary,
        "uses_test_split": False,
        "uses_semantic_granularity_supervision": False,
        "saved_checkpoint_type": "best",
    }

    if as_bool(config.get("save_last_checkpoint", True)):
        last_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.last{checkpoint_path.suffix}")
        adapter.load_state_dict(final_state)
        last_metadata = {**metadata, "saved_checkpoint_type": "last"}
        save_adapter_checkpoint(adapter, str(last_path), last_metadata)
        last_path.with_suffix(".json").write_text(
            json.dumps(last_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata["last_checkpoint"] = str(last_path)

    adapter.load_state_dict(best_state)
    save_adapter_checkpoint(adapter, str(checkpoint_path), metadata)
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best adapter checkpoint: {checkpoint_path}")
    print(f"Saved adapter metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Hyperbolic_adapter_coco.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda_id", default=0, type=int)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--adapter_checkpoint_name", default=None)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--source_text_encoder", default=None)
    parser.add_argument("--albef_ckpt", default=None)
    parser.add_argument("--batch_size_train", default=None, type=int)
    parser.add_argument("--hier_feature_layers", default=None)
    parser.add_argument("--clip_model_path", default=None)
    parser.add_argument("--cnn_token_grid_size", default=None, type=int)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--validation_max_images", default=None, type=int)
    parser.add_argument("--semantic_probe_max_images", default=None, type=int)
    parser.add_argument("--semantic_probe_batch_size", default=None, type=int)
    parser.add_argument("--spacy_model", default=None)
    parser.add_argument("--skip_semantic_probe", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.load(handle)
    overrides = {
        "max_train_batches": args.max_train_batches,
        "output_dir": args.output_dir,
        "adapter_checkpoint_name": args.adapter_checkpoint_name,
        "num_workers": args.num_workers,
        "source_text_encoder": args.source_text_encoder,
        "albef_ckpt": args.albef_ckpt,
        "batch_size_train": args.batch_size_train,
        "hier_feature_layers": args.hier_feature_layers,
        "clip_model_path": args.clip_model_path,
        "cnn_token_grid_size": args.cnn_token_grid_size,
        "val_file": args.val_file,
        "validation_max_images": args.validation_max_images,
        "semantic_probe_max_images": args.semantic_probe_max_images,
        "semantic_probe_batch_size": args.semantic_probe_batch_size,
        "spacy_model": args.spacy_model,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.epochs is not None:
        config.setdefault("schedular", {})["epochs"] = args.epochs
    if args.skip_semantic_probe:
        config["semantic_probe_enabled"] = False
    train(args, config)


if __name__ == "__main__":
    main()
