import argparse
import os

from ruamel.yaml import YAML

yaml=YAML(typ='safe')
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip

import utils
import copy
import time

from SA_AET import Attacker, ImageAttackObjectiveConfig, ImageAttacker, TextAttacker
from dataset import paired_dataset2
from hierarchical_visual_attack import HyperbolicGuidanceConfig

def toImage(norm_img):
    pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
    pil_img=Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
    return pil_img

def parse_hier_feature_layers(layer_arg, source_model="CLIP_ViT"):
    """将命令行层配置转为视觉编码器中的真实层名。"""

    if layer_arg is None:
        return None

    feature_layers = []
    for item in layer_arg.split(','):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            if str(source_model) in {"ALBEF", "TCL"}:
                feature_layers.append(f"blocks.{item}")
            else:
                feature_layers.append(f"transformer.resblocks.{item}")
        else:
            feature_layers.append(item)
    return feature_layers or None

def resolve_visual_sequence_first(source_model):
    """根据源模型确定 hooked visual tokens 是否为 `[N,B,C]`。"""

    return str(source_model) not in {"ALBEF", "TCL"}

def resolve_attack_objective(args):
    """解析新的图像侧攻击目标，并兼容旧双曲入口。"""

    if args.attack_objective is not None:
        return args

    if args.auxiliary_loss_type == "euclidean_token_caption" or args.enable_hier_visual:
        raise ValueError(
            "euclidean_token_caption / --enable_hier_visual is deprecated in the "
            "main method. Use --attack_objective baseline|original_plus_hyperbolic|"
            "hyperbolic_main."
        )
    if args.auxiliary_loss_type == "hyperbolic_adapter" or args.enable_hyperbolic_guidance:
        args.attack_objective = (
            "hyperbolic_main"
            if args.image_loss_composition == "hyperbolic_main"
            else "original_plus_hyperbolic"
        )
    elif args.auxiliary_loss_type == "none":
        args.attack_objective = "baseline"
    else:
        args.attack_objective = "original_plus_hyperbolic"
    return args

def freeze_eval_module(module):
    """Freeze eval-only modules while keeping input gradients available."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module

def prepare_eval_module(module, device, freeze=False, cast_float=False):
    """Move a module to device and apply optional eval-time freezing."""

    module.to(device)
    if cast_float:
        module.float()
    module.eval()
    if freeze:
        freeze_eval_module(module)
    return module

def fill_source_feature_cache(
    model,
    tokenizer,
    images_normalize,
    adv_images,
    adv_texts,
    images_ids,
    texts_ids,
    s_feat_dict,
    source_model,
    device,
):
    """Write adversarial source-model features for one batch into CPU cache."""

    s_adv_images_norm = images_normalize(adv_images)
    if source_model in ['ALBEF', 'TCL']:
        adv_texts_input = tokenizer(
            adv_texts,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        s_output_img = model.inference_image(s_adv_images_norm)
        s_output_txt = model.inference_text(adv_texts_input)

        s_feat_dict['s_image_feats'][images_ids] = s_output_img['image_feat'].cpu().detach()
        s_feat_dict['s_image_embeds'][images_ids] = s_output_img['image_embed'].cpu().detach()
        s_feat_dict['s_text_feats'][texts_ids] = s_output_txt['text_feat'].cpu().detach()
        s_feat_dict['s_text_embeds'][texts_ids] = s_output_txt['text_embed'].cpu().detach()
        s_feat_dict['s_text_atts'][texts_ids] = adv_texts_input.attention_mask.cpu().detach()
    else:
        output = model.inference(s_adv_images_norm, adv_texts)
        s_feat_dict['s_image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
        s_feat_dict['s_text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()

def fill_target_feature_cache(
    t_model_name,
    t_model,
    tokenizer,
    images_normalize,
    t_test_transform,
    adv_images,
    adv_texts,
    images_ids,
    texts_ids,
    t_feat_dict,
    device,
):
    """Write adversarial target-model features for one batch into CPU cache."""

    t_adv_img_list = []
    for itm in adv_images:
        t_adv_img_list.append(t_test_transform(itm))
    t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)
    t_adv_images_norm = images_normalize(t_adv_imgs)
    if t_model_name in ['ALBEF', 'TCL']:
        adv_texts_input = tokenizer(
            adv_texts,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        t_output_img = t_model.inference_image(t_adv_images_norm)
        t_output_txt = t_model.inference_text(adv_texts_input)
        t_feat_dict['t_image_feats'][images_ids] = t_output_img['image_feat'].cpu().detach()
        t_feat_dict['t_image_embeds'][images_ids] = t_output_img['image_embed'].cpu().detach()
        t_feat_dict['t_text_feats'][texts_ids] = t_output_txt['text_feat'].cpu().detach()
        t_feat_dict['t_text_embeds'][texts_ids] = t_output_txt['text_embed'].cpu().detach()
        t_feat_dict['t_text_atts'][texts_ids] = adv_texts_input.attention_mask.cpu().detach()
    else:
        output = t_model.inference(t_adv_images_norm, adv_texts)
        t_feat_dict['t_image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
        t_feat_dict['t_text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()

def score_retrieval_for_model(model_name, model, feat_dict, num_image, num_text, device):
    """Compute retrieval score matrices for source or target feature caches."""

    if model_name in ['ALBEF', 'TCL']:
        prefix = 's' if 's_image_feats' in feat_dict else 't'
        score_matrix_i2t, score_matrix_t2i = retrieval_score(
            model,
            feat_dict[f'{prefix}_image_feats'],
            feat_dict[f'{prefix}_image_embeds'],
            feat_dict[f'{prefix}_text_feats'],
            feat_dict[f'{prefix}_text_embeds'],
            feat_dict[f'{prefix}_text_atts'],
            num_image,
            num_text,
            device=device,
        )
        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    prefix = 's' if 's_image_feats' in feat_dict else 't'
    sims_matrix = feat_dict[f'{prefix}_image_feats'] @ feat_dict[f'{prefix}_text_feats'].t()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()

def retrieval_eval(model, ref_model, t_models, t_ref_models, t_test_transforms, data_loader, tokenizer, t_tokenizers, device, args,config):
    prepare_eval_module(
        model,
        device,
        freeze=args.freeze_eval_models,
        cast_float=True,
    )
    prepare_eval_module(
        ref_model,
        device,
        freeze=args.freeze_eval_models,
    )

    if args.lazy_target_models:
        for t_model, t_ref_model in zip(t_models, t_ref_models):
            t_model.float()
            t_model.eval()
            t_ref_model.eval()
            if args.freeze_eval_models:
                freeze_eval_module(t_model)
                freeze_eval_module(t_ref_model)
    else:
        for t_model, t_ref_model in zip(t_models, t_ref_models):
            prepare_eval_module(
                t_model,
                device,
                freeze=args.freeze_eval_models,
                cast_float=True,
            )
            prepare_eval_module(
                t_ref_model,
                device,
                freeze=args.freeze_eval_models,
            )

    print('Computing features for evaluation adv...')

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    objective_config = ImageAttackObjectiveConfig(
        attack_objective=args.attack_objective,
        lambda_hyp=args.lambda_hyp,
    )
    hyperbolic_config = HyperbolicGuidanceConfig(
        lambda_hyp=args.lambda_hyp,
        adapter_checkpoint=args.adapter_checkpoint,
        adapter_dim=args.adapter_dim,
        adapter_input_normalization=args.adapter_input_normalization,
        adapter_train_data=args.adapter_train_data,
        adapter_train_objective=args.adapter_train_objective,
        adapter_freeze_in_attack=True,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        curvature=args.hyperbolic_curvature,
        eps=args.hyperbolic_eps,
        alignment_temperature=args.alignment_temperature,
        radius_temperature=args.radius_temperature,
        layer_weight_strategy=args.layer_weight_strategy,
        hard_negative_topk=args.hard_negative_topk,
        ranking_temperature=args.ranking_temperature,
        attack_loss_mode=args.hyp_attack_loss_mode,
        gradient_fusion_strategy=args.hyp_gradient_fusion_strategy,
        geometry_mode=args.hyp_geometry_mode,
        use_radius_gap=not args.disable_radius_gap,
    )
    img_attacker = ImageAttacker(
        images_normalize,
        eps=args.image_eps / 255,
        steps=args.attack_steps,
        step_size=args.image_step_size / 255,
        sample_numbers=args.image_sample_numbers,
        attack_objective_config=objective_config,
        hyperbolic_config=hyperbolic_config,
    )

    max_length = 30 if args.source_model in ['ALBEF', 'TCL'] else 77 
    txt_attacker = TextAttacker(
        ref_model,
        tokenizer,
        cls=False,
        max_length=max_length,
        number_perturbation=args.text_num_perturbation,
        topk=args.text_topk,
        threshold_pred_score=args.text_threshold_pred_score,
        text_ratios=[float(item) for item in args.text_ratios.split(',')],
    )
    visual_sequence_first = resolve_visual_sequence_first(args.source_model)
    attacker = Attacker(
        model,
        img_attacker,
        txt_attacker,
        feature_layers=parse_hier_feature_layers(args.hier_feature_layers, args.source_model),
        token_mode=args.token_granularity,
        sequence_first=visual_sequence_first,
        cnn_token_grid_size=args.cnn_token_grid_size,
    )
    print(
        "Attack modules: "
        f"attack_objective={args.attack_objective}, "
        f"lambda_hyp={args.lambda_hyp}, "
        f"adapter_checkpoint={args.adapter_checkpoint}, "
        f"adapter_dim={args.adapter_dim}, "
        f"adapter_train_data={args.adapter_train_data}, "
        f"adapter_train_objective={args.adapter_train_objective}, "
        f"adapter_freeze_in_attack=True, "
        f"adapter_input_normalization={args.adapter_input_normalization}, "
        f"alignment_temperature={args.alignment_temperature}, "
        f"radius_temperature={args.radius_temperature}, "
        f"hard_negative_topk={args.hard_negative_topk}, "
        f"ranking_temperature={args.ranking_temperature}, "
        f"hyp_attack_loss_mode={args.hyp_attack_loss_mode}, "
        f"hyp_gradient_fusion_strategy={args.hyp_gradient_fusion_strategy}, "
        f"hyp_geometry_mode={args.hyp_geometry_mode}, "
        f"use_radius_gap={not args.disable_radius_gap}, "
        f"radius_min={args.radius_min}, "
        f"radius_max={args.radius_max}, "
        f"hyperbolic_tau={args.hyperbolic_tau}, "
        f"token_granularity={args.token_granularity}, "
        f"visual_sequence_first={visual_sequence_first}, "
        f"cnn_token_grid_size={args.cnn_token_grid_size}, "
        f"feature_layers={args.hier_feature_layers}, "
        f"attack_steps={args.attack_steps}, "
        f"image_eps={args.image_eps}, "
        f"image_step_size={args.image_step_size}, "
        f"image_sample_numbers={args.image_sample_numbers}, "
        f"text_num_perturbation={args.text_num_perturbation}, "
        f"text_topk={args.text_topk}, "
        f"text_threshold_pred_score={args.text_threshold_pred_score}, "
        f"text_ratios={args.text_ratios}"
    )

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    s_feat_dict = {}
    if args.source_model in ['ALBEF', 'TCL']:
        s_feat_dict['s_image_feats'] = torch.zeros(num_image, config['embed_dim'])
        s_feat_dict['s_image_embeds'] = torch.zeros(num_image, 577, 768)
        s_feat_dict['s_text_feats'] = torch.zeros(num_text, config['embed_dim'])
        s_feat_dict['s_text_embeds'] = torch.zeros(num_text, 30, 768)
        s_feat_dict['s_text_atts'] = torch.zeros(num_text, 30).long()
    else:
        s_feat_dict['s_image_feats'] = torch.zeros(num_image, model.visual.output_dim)
        s_feat_dict['s_text_feats'] = torch.zeros(num_text, model.visual.output_dim)

    t_feat_dicts = []
    t_model_names = copy.deepcopy(args.model_list)
    t_model_names.remove(args.source_model)
    for t_model_name,t_model in zip(t_model_names,t_models):
        t_feat_dict = {}
        if t_model_name in ['ALBEF', 'TCL']:
            t_feat_dict['t_image_feats'] = torch.zeros(num_image, config['embed_dim'])
            t_feat_dict['t_image_embeds'] = torch.zeros(num_image, 577, 768)
            t_feat_dict['t_text_feats'] = torch.zeros(num_text, config['embed_dim'])
            t_feat_dict['t_text_embeds'] = torch.zeros(num_text, 30, 768)
            t_feat_dict['t_text_atts'] = torch.zeros(num_text, 30).long()
        else:
            t_feat_dict['t_image_feats'] = torch.zeros(num_image, t_model.visual.output_dim)
            t_feat_dict['t_text_feats'] = torch.zeros(num_text, t_model.visual.output_dim)
        t_feat_dicts.append(t_feat_dict)

    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)
    else:
        scales = None

    print('Forward')

    all_texts_all=[]
    adv_batch_cache = []

    for batch_idx, (images, texts_group, images_ids, text_ids_groups,_) in enumerate(data_loader):
        if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
            break
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')
        for index_text in range(len(texts_group)):
            all_texts_all+=texts_group[index_text]

    num_samples = max(1, int(0.4 * len(all_texts_all))) if all_texts_all else 0

       #使用这些索引来选取张量中的数据
    all_texts = random.sample(all_texts_all, num_samples)

    for batch_idx, (images, texts_group, images_ids, text_ids_groups,image_paths) in enumerate(data_loader):
        if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
            break
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')
        texts_ids = []
        txt2img = []
        texts = []
        for i in range(len(texts_group)):
            texts += texts_group[i]
            texts_ids += text_ids_groups[i]
            txt2img += [i]*len(text_ids_groups[i])

        images = images.to(device)

        adv_images, adv_texts,execuate_time = attacker.attack(images, texts, txt2img,all_texts, device=device,
                                                max_length=max_length, scales=scales)

        with torch.no_grad():
            fill_source_feature_cache(
                model,
                tokenizer,
                images_normalize,
                adv_images,
                adv_texts,
                images_ids,
                texts_ids,
                s_feat_dict,
                args.source_model,
                device,
            )

            if args.lazy_target_models:
                adv_batch_cache.append({
                    "adv_images": adv_images.detach().cpu(),
                    "adv_texts": list(adv_texts),
                    "images_ids": list(images_ids),
                    "texts_ids": list(texts_ids),
                })
            else:
                for t_model_name,t_model,t_feat_dict,t_test_transform in zip(t_model_names,t_models,t_feat_dicts,t_test_transforms):
                    fill_target_feature_cache(
                        t_model_name,
                        t_model,
                        tokenizer,
                        images_normalize,
                        t_test_transform,
                        adv_images,
                        adv_texts,
                        images_ids,
                        texts_ids,
                        t_feat_dict,
                        device,
                    )

    if args.max_eval_batches is not None:
        args.latest_hyperbolic_stats = getattr(
            img_attacker,
            "latest_hyperbolic_stats",
            {},
        )
        args.latest_image_loss_stats = getattr(
            img_attacker,
            "latest_image_loss_stats",
            {},
        )
        print("Smoke run finished before full retrieval scoring because --max_eval_batches is set.")
        return None, None, [], []

    s_score_matrix_i2t, s_score_matrix_t2i = score_retrieval_for_model(
        args.source_model,
        model,
        s_feat_dict,
        num_image,
        num_text,
        device,
    )
    
    t_score_matrix_i2ts= [] 
    t_score_matrix_t2is= []
    if args.lazy_target_models:
        model.to('cpu')
        ref_model.to('cpu')
        torch.cuda.empty_cache()

        for t_model_name,t_feat_dict,t_model,t_test_transform in zip(t_model_names,t_feat_dicts,t_models,t_test_transforms):
            prepare_eval_module(
                t_model,
                device,
                freeze=args.freeze_eval_models,
                cast_float=True,
            )
            with torch.no_grad():
                for adv_batch in adv_batch_cache:
                    fill_target_feature_cache(
                        t_model_name,
                        t_model,
                        tokenizer,
                        images_normalize,
                        t_test_transform,
                        adv_batch["adv_images"],
                        adv_batch["adv_texts"],
                        adv_batch["images_ids"],
                        adv_batch["texts_ids"],
                        t_feat_dict,
                        device,
                    )
            t_score_matrix_i2t, t_score_matrix_t2i = score_retrieval_for_model(
                t_model_name,
                t_model,
                t_feat_dict,
                num_image,
                num_text,
                device,
            )
            t_score_matrix_i2ts.append(t_score_matrix_i2t)
            t_score_matrix_t2is.append(t_score_matrix_t2i)
            t_model.to('cpu')
            torch.cuda.empty_cache()
    else:
        for t_model_name,t_feat_dict,t_model in zip(t_model_names,t_feat_dicts,t_models):
            t_score_matrix_i2t, t_score_matrix_t2i = score_retrieval_for_model(
                t_model_name,
                t_model,
                t_feat_dict,
                num_image,
                num_text,
                device,
            )
            t_score_matrix_i2ts.append(t_score_matrix_i2t)
            t_score_matrix_t2is.append(t_score_matrix_t2i)

    args.latest_hyperbolic_stats = getattr(
        img_attacker,
        "latest_hyperbolic_stats",
        {},
    )
    args.latest_image_loss_stats = getattr(
        img_attacker,
        "latest_image_loss_stats",
        {},
    )
    return s_score_matrix_i2t, s_score_matrix_t2i, \
        t_score_matrix_i2ts, t_score_matrix_t2is

@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)


    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    
    original_rank_index_path = args.original_rank_index_path
    origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2) 
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]


    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)


    eval_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})',
                   'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})',
                   'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})',
                   'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})',
                   'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})',
                   'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}
    return eval_result

def load_model(args,model_name,text_encoder, device, initial_device=None):
    initial_device = initial_device or device
    # tokenizer = BertTokenizer.from_pretrained(text_encoder)
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        model_ckpt = args.albef_ckpt if model_name == 'ALBEF' else args.tcl_ckpt
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    ### load checkpoint
    else:
        prefix = '/home/data/share_data/multimodal/models/clip/'
        model_name = 'ViT-B-16.pt' if model_name == 'CLIP_ViT' else 'RN101.pt'
        model_name = prefix + model_name
        model, preprocess = clip.load(model_name, device=initial_device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer

def eval_asr(model, ref_model, tokenizer, t_models, t_ref_models, t_tokenizers, t_test_transforms, data_loader, device, args, config):
    print("Start eval")
    start_time = time.time()
    
    score_i2t, score_t2i, t_score_i2ts, t_score_t2is = retrieval_eval(model, ref_model, t_models, t_ref_models, t_test_transforms,
                                                                   data_loader, tokenizer, t_tokenizers, device, args,config)

    if args.max_eval_batches is not None:
        if getattr(args, "latest_image_loss_stats", None):
            print(
                "Image loss stats: "
                f"{json.dumps(args.latest_image_loss_stats, ensure_ascii=False)}"
            )
        if getattr(args, "latest_hyperbolic_stats", None):
            print(
                "Hyperbolic stats: "
                f"{json.dumps(args.latest_hyperbolic_stats, ensure_ascii=False)}"
            )
        print("Skip ASR metrics for smoke run with incomplete feature cache.")
        return


    result_file_path = args.result_file_path

    with open(result_file_path, "a") as file:
        file.write("\n") 
        file.write(
            "Run config: "
            f"source_model={args.source_model}, "
            f"attack_objective={args.attack_objective}, "
            f"lambda_hyp={args.lambda_hyp}, "
            f"adapter_checkpoint={args.adapter_checkpoint}, "
            f"adapter_dim={args.adapter_dim}, "
            f"adapter_train_data={args.adapter_train_data}, "
            f"adapter_train_objective={args.adapter_train_objective}, "
            f"adapter_freeze_in_attack=True, "
            f"adapter_input_normalization={args.adapter_input_normalization}, "
            f"alignment_temperature={args.alignment_temperature}, "
            f"radius_temperature={args.radius_temperature}, "
            f"hard_negative_topk={args.hard_negative_topk}, "
            f"ranking_temperature={args.ranking_temperature}, "
            f"hyp_attack_loss_mode={args.hyp_attack_loss_mode}, "
            f"hyp_gradient_fusion_strategy={args.hyp_gradient_fusion_strategy}, "
            f"hyp_geometry_mode={args.hyp_geometry_mode}, "
            f"use_radius_gap={not args.disable_radius_gap}, "
            f"radius_min={args.radius_min}, "
            f"radius_max={args.radius_max}, "
            f"hyperbolic_tau={args.hyperbolic_tau}, "
            f"hyperbolic_curvature={args.hyperbolic_curvature}, "
            f"token_granularity={args.token_granularity}, "
            f"visual_sequence_first={resolve_visual_sequence_first(args.source_model)}, "
            f"cnn_token_grid_size={args.cnn_token_grid_size}, "
            f"layer_weight_strategy={args.layer_weight_strategy}, "
            f"hier_feature_layers={args.hier_feature_layers}, "
            f"attack_steps={args.attack_steps}, "
            f"image_eps={args.image_eps}, "
            f"image_step_size={args.image_step_size}, "
            f"image_sample_numbers={args.image_sample_numbers}, "
            f"text_num_perturbation={args.text_num_perturbation}, "
            f"text_topk={args.text_topk}, "
            f"text_threshold_pred_score={args.text_threshold_pred_score}, "
            f"text_ratios={args.text_ratios}\n"
        )
        if getattr(args, "latest_image_loss_stats", None):
            file.write(
                "Image loss stats: "
                f"{json.dumps(args.latest_image_loss_stats, ensure_ascii=False)}\n"
            )
        if getattr(args, "latest_hyperbolic_stats", None):
            file.write(
                "Hyperbolic stats: "
                f"{json.dumps(args.latest_hyperbolic_stats, ensure_ascii=False)}\n"
            )
        result = itm_eval(score_i2t, score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, args.source_model)
        file.write("Performance on {}: \n {}".format(args.source_model, result) + "\n")

        t_model_names = copy.deepcopy(args.model_list)
        t_model_names.remove(args.source_model)
        for t_model_name, t_score_i2t, t_score_t2i in zip(t_model_names, t_score_i2ts, t_score_t2is):
            t_result = itm_eval(t_score_i2t, t_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, t_model_name)
            file.write("Performance on {}: \n {}".format(t_model_name, t_result) + "\n")
    
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))

def main(args, config):
    torch.cuda.set_device(args.cuda_id)
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(args,args.source_model,args.source_text_encoder, device)

    print("Creating Target Model")
    t_models = []
    t_ref_models = []
    t_tokenizers = []
    t_model_names = copy.deepcopy(args.model_list)
    t_model_names.remove(args.source_model)
    for t_model_name in t_model_names:
        target_initial_device = 'cpu' if args.lazy_target_models else device
        t_model, t_ref_model, t_tokenizer = load_model(
            args,
            t_model_name,
            args.target_text_encoder,
            device,
            initial_device=target_initial_device,
        )
        t_models.append(t_model)
        t_ref_models.append(t_ref_model)
        t_tokenizers.append(t_tokenizer)
   
    #### Dataset ####
    print("Creating dataset")
    
    s_test_transform = None
    if args.source_model in ['ALBEF', 'TCL']:
        s_test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),        
        ])
    else:
        n_px = model.visual.input_resolution
        s_test_transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),       
        ])

    t_test_transforms = []
    for index,t_model_name in enumerate(t_model_names):
        if t_model_name in ['ALBEF', 'TCL']:
            t_test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
                transforms.ToTensor(),  
            ])
            t_test_transforms.append(t_test_transform)
        else:
            t_model = t_models[index]
            t_n_px = t_model.visual.input_resolution
            t_test_transform = transforms.Compose([
                # transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Resize(t_n_px, interpolation=Image.BICUBIC),
                transforms.CenterCrop(t_n_px),
                # transforms.ToTensor(),
            ])
            t_test_transforms.append(t_test_transform)
    
    test_dataset = paired_dataset2(config['test_file'], s_test_transform, config['image_root'])
    # indices = list(range(min(10, len(test_dataset))))
    # test_dataset = utils.AttributePreservingSubset(test_dataset, indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, collate_fn=test_dataset.collate_fn)

    eval_asr(model, ref_model, tokenizer, t_models, t_ref_models, t_tokenizers, t_test_transforms, test_loader, device, args, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr_re.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--cuda_id', default=1, type=int)

    parser.add_argument('--model_list', default=['CLIP_ViT', 'ALBEF', 'TCL', 'CLIP_CNN'], type=list)
    parser.add_argument('--source_model', default='CLIP_ViT', type=str)
    parser.add_argument('--source_text_encoder', default='/home/share_dir/from_jyg/tokenizer/bert-base-uncased', type=str)
    parser.add_argument('--target_text_encoder', default='/home/share_dir/from_jyg/tokenizer/bert-base-uncased', type=str)

    parser.add_argument('--albef_ckpt', default='/home/data/share_data/multimodal/models/ALBEF/albef_flickr30k.pth', type=str) 
    parser.add_argument('--tcl_ckpt', default='/home/data/share_data/multimodal/models/TCL/checkpoint_flickr_finetune.pth', type=str)    
 
    parser.add_argument('--original_rank_index_path', default='./std_eval_idx/flickr30k/')  
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    parser.add_argument(
        '--attack_objective',
        default=None,
        choices=['baseline', 'original_plus_hyperbolic', 'hyperbolic_main'],
        help='图像侧攻击目标：baseline / original_plus_hyperbolic / hyperbolic_main。',
    )
    parser.add_argument('--enable_hier_visual', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lambda_hier', default=0.05, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--token_granularity', default='patch', choices=['cls', 'patch', 'cls_patch'])
    parser.add_argument('--token_attention_tau', default=0.1, type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        '--token_aggregation',
        default='caption_attention',
        choices=['caption_attention', 'uniform'],
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--layer_weight_strategy', default='fixed_depth', choices=['uniform', 'fixed_depth', 'late_only'])
    parser.add_argument('--hier_feature_layers', default='2,5,8,11', type=str)
    parser.add_argument('--cnn_token_grid_size', default=None, type=int)
    parser.add_argument('--auxiliary_loss_type', default=None, choices=['none', 'euclidean_token_caption', 'hyperbolic_adapter'], help=argparse.SUPPRESS)
    parser.add_argument('--enable_hyperbolic_guidance', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lambda_hyp', default=0.03, type=float)
    parser.add_argument('--adapter_checkpoint', default=None, type=str)
    parser.add_argument('--adapter_dim', default=256, type=int)
    parser.add_argument('--adapter_train_data', default='mscoco_train', choices=['mscoco_train', 'flickr30k_train', 'imagenet_templates'])
    parser.add_argument('--adapter_train_objective', default='clean_ranking_logmeanexp', choices=['clean_ranking_logmeanexp'])
    parser.add_argument('--adapter_input_normalization', default='layernorm_l2', choices=['none', 'l2', 'layernorm_l2'])
    parser.add_argument('--ranking_temperature', default=0.1, type=float)
    parser.add_argument('--hard_negative_topk', default=5, type=int)
    parser.add_argument('--hyp_attack_loss_mode', default='positive_distance', choices=['ranking', 'positive_distance', 'hybrid'])
    parser.add_argument(
        '--hyp_gradient_fusion_strategy',
        default='pcgrad_norm',
        choices=['additive', 'normalized_additive', 'pcgrad_norm', 'candidate_guided'],
    )
    parser.add_argument('--hyp_geometry_mode', default='hyperbolic', choices=['hyperbolic', 'euclidean'])
    parser.add_argument('--disable_radius_gap', action='store_true')
    parser.add_argument('--alignment_temperature', default=None, type=float)
    parser.add_argument('--radius_temperature', default=None, type=float)
    parser.add_argument(
        '--image_loss_composition',
        default='original_plus_aux',
        choices=['original_plus_aux', 'hyperbolic_main'],
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--hyperbolic_tau', default=0.1, type=float)
    parser.add_argument('--hyperbolic_curvature', default=1.0, type=float)
    parser.add_argument('--hyperbolic_eps', default=1e-5, type=float)
    parser.add_argument('--radius_min', default=0.1, type=float)
    parser.add_argument('--radius_max', default=0.8, type=float)
    parser.add_argument('--max_eval_batches', default=None, type=int)
    parser.add_argument('--attack_steps', default=10, type=int)
    parser.add_argument('--image_eps', default=8.0, type=float)
    parser.add_argument('--image_step_size', default=2.0, type=float)
    parser.add_argument('--image_sample_numbers', default=5, type=int)
    parser.add_argument('--text_num_perturbation', default=1, type=int)
    parser.add_argument('--text_topk', default=10, type=int)
    parser.add_argument('--text_threshold_pred_score', default=0.3, type=float)
    parser.add_argument('--text_ratios', default='0.6,0.2,0.2', type=str)
    parser.add_argument('--freeze_eval_models', action='store_true')
    parser.add_argument('--lazy_target_models', action='store_true')
    parser.add_argument('--result_file_path', default='./result_AET.txt', type=str)
    args = parser.parse_args()
    if args.alignment_temperature is None:
        args.alignment_temperature = args.hyperbolic_tau
    if args.radius_temperature is None:
        args.radius_temperature = args.alignment_temperature
    try:
        args = resolve_attack_objective(args)
    except ValueError as exc:
        parser.error(str(exc))

    config = yaml.load(open(args.config, 'r'))
    result_dir = os.path.dirname(args.result_file_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    main(args, config)    
