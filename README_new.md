# HySIA: Hyperbolic Semantic Image–Text Association for Transferable Attacks against VLMs

Official PyTorch implementation of **HySIA**, a transferable multimodal adversarial attack that combines hierarchical visual supervision with structured image–text association in a Poincaré ball.

**Paper:** [Jia.pdf](references/Jia.pdf)  
**Authors:** Yonggang Jia, Kaiyu Wang, Guangsheng Feng, Hongwu Lv, Sulin Gao, and Wenqi Zheng  
**Institution:** College of Computer Science and Technology, Harbin Engineering University

> This repository is intended for authorized adversarial-robustness research. Use it only on models, data, and systems that you own or are permitted to evaluate.

## Abstract

Transferable adversarial examples expose vulnerabilities shared by different vision–language models (VLMs). Existing attacks usually optimize a flat Euclidean image–text objective using only final-layer visual representations. This can discard useful intermediate visual cues and make the attack overfit the surrogate model.

HySIA addresses these limitations with two components:

- **Hierarchical Visual Feature Extraction and Projection Module (HEPM):** extracts complementary patch-level representations from multiple visual-encoder depths and maps them, together with text representations, into a shared Poincaré ball.
- **Hyperbolic Image–Text Association Module (HITAM):** combines hyperbolic proximity and radial semantic-granularity discrepancy to construct structured token–caption associations for adversarial optimization.

The resulting objective augments the original semantic-aligned image–text loss:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{sim}}
+
\lambda_{\mathrm{hyp}}\mathcal{L}_{\mathrm{hyp}}.
$$

The source encoder and the offline-trained hyperbolic adapter remain frozen during attack generation; only the image pixels are updated by projected gradient descent (PGD). Text perturbations are generated with BERT-Attack in the alternating image–text attack pipeline.

<p align="center">
  <img src="paper/2026_ICASSP_paper/figs/Jia-fig1v2.png" width="82%" alt="Motivation for HySIA">
</p>

## Method at a glance

1. Generate an initial adversarial caption with BERT-Attack.
2. Extract patch tokens from several depths of the frozen surrogate visual encoder.
3. Project visual tokens and caption embeddings into a shared Poincaré ball with a source-specific adapter trained offline on clean MSCOCO image–text pairs.
4. Compute token–caption hyperbolic distance and radial discrepancy, then aggregate the resulting structured associations into the HITAM loss.
5. Combine the HITAM loss with the original SA-AET image objective and update the image under an $\ell_\infty$ constraint.
6. Regenerate the adversarial caption using clean, previous, and current adversarial image features.
7. Evaluate the final adversarial image–text pair on unseen VLMs or downstream tasks.

The current implementation supports three image-loss routes:

| `--attack_objective` | Objective |
|---|---|
| `baseline` | Original SA-AET image objective only |
| `original_plus_hyperbolic` | $\mathcal{L}_{\mathrm{sim}} + \lambda_{\mathrm{hyp}}\mathcal{L}_{\mathrm{hyp}}$ (HySIA) |
| `hyperbolic_main` | Hyperbolic objective only |

## Main results reported in the paper

### Cross-model transfer on Flickr30K

Entries are **TR R@1 ASR / IR R@1 ASR (%)**. Diagonal entries are white-box results; all other entries are black-box transfers.

| Source \ Target | ALBEF | TCL | CLIP-ViT | CLIP-CNN |
|---|---:|---:|---:|---:|
| ALBEF | **99.90 / 99.95** | 95.68 / 95.88 | 57.79 / 65.40 | 60.41 / 67.28 |
| TCL | 98.23 / 98.34 | **100.00 / 100.00** | 54.48 / 64.11 | 60.41 / 67.99 |
| CLIP-ViT | 39.52 / 52.48 | 42.36 / 53.12 | **100.00 / 99.97** | 70.11 / 75.85 |
| CLIP-CNN | 24.71 / 38.40 | 26.77 / 41.88 | 55.46 / 63.37 | **100.00 / 99.86** |

HySIA obtains an average off-diagonal black-box ASR of **61.25%** across the four source models and both retrieval directions.

### Cross-task transfer to visual grounding

Adversarial examples are generated on ALBEF and evaluated on RefCOCO+. Lower grounding accuracy indicates a stronger attack.

| Split | Clean | HySIA |
|---|---:|---:|
| Val | 58.44 | **37.90** |
| TestA | 65.91 | **41.50** |
| TestB | 46.25 | **31.60** |

### Module ablation with ALBEF as the source

The mean below is computed over the six black-box TR/IR R@1 entries for TCL, CLIP-ViT, and CLIP-CNN.

| Method | Mean black-box ASR (%) |
|---|---:|
| SA-AET | 72.45 |
| w/o HEPM | 72.86 |
| w/o HITAM | 72.64 |
| **HySIA** | **73.74** |

## Paper configuration

The following settings reproduce the protocol described in the supplied paper. Source-specific adapter layers must match the checkpoint metadata.

| Setting | Value |
|---|---|
| Evaluation datasets | Flickr30K retrieval; RefCOCO+ visual grounding |
| Surrogate/target VLMs | ALBEF, TCL, CLIP ViT-B/16, CLIP RN101 |
| PGD budget | $\epsilon=8/255$ |
| PGD iterations / step size | 10 / $2/255$ |
| Text perturbations / candidates | 1 / 10 |
| Hyperbolic curvature | 1.0 |
| Adapter output dimension | 256 |
| Radius interval | $[0.1, 0.8]$ |
| Distance / radius / ranking temperatures | 0.1 / 0.1 / 0.1 |
| Hyperbolic weight | $\lambda_{\mathrm{hyp}}=0.2$ |
| Offline adapter data | MSCOCO training split |
| Attack-time adapter state | Frozen |

## Repository structure

```text
.
├── SA_AET.py                         # alternating text/image attack and objective fusion
├── eval_AET.py                       # Flickr30K attack and cross-model evaluation
├── eval_AET_albef_mscoco.py          # MSCOCO-oriented evaluation entry point
├── train_hyperbolic_adapter.py        # offline adapter training and validation
├── hyperbolic.py                      # Poincaré-ball operations
├── hierarchical_visual_attack/
│   ├── vision_extractor.py            # multi-depth visual feature hooks (HEPM)
│   ├── hyperbolic_adapter.py           # learnable projection adapter (HEPM)
│   ├── hyperbolic_guidance.py          # association weights and attack losses (HITAM)
│   ├── attack_loss.py
│   └── relation_mapper.py
├── configs/                           # retrieval, grounding, and adapter configs
├── models/                            # ALBEF and OpenAI CLIP model definitions
├── scripts/                           # training, evaluation, and ablation launchers
├── data_annotation/                   # dataset annotations (images are not included)
├── std_eval_idx/                      # clean retrieval rank indices used for ASR
├── tests/                             # unit tests for HEPM/HITAM and adapter training
└── references/Jia.pdf                 # paper used by this README
```

Legacy SA-AET/SGA/DRA files are retained for baseline compatibility. The HySIA path is implemented by `SA_AET.py`, `hierarchical_visual_attack/`, `train_hyperbolic_adapter.py`, and the HySIA launchers under `scripts/`.

## Installation

The local development environment used for this repository is Python 3.8 with PyTorch 2.1.0, torchvision 0.16.0, transformers 4.28.1, timm 0.6.13, and spaCy 3.7.5. Install the PyTorch build that matches your CUDA runtime; for CUDA 11.8:

```bash
conda create -n hysia python=3.8 -y
conda activate hysia

pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install pytest
```

## Data and base-model preparation

Download datasets and model weights from their official projects:

- [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)
- [MSCOCO](https://cocodataset.org/#download)
- [ALBEF](https://github.com/salesforce/ALBEF)
- [TCL](https://github.com/uta-smile/TCL)
- [OpenAI CLIP](https://github.com/openai/CLIP)

A convenient local layout is:

```text
datasets/
├── flickr30k-images/
└── coco/
    ├── train2014/
    └── val2014/

checkpoints/
├── base_models/
│   ├── albef_flickr30k.pth
│   ├── tcl_flickr30k.pth
│   ├── ViT-B-16.pt
│   └── RN101.pt
└── hyperbolic_adapter/
    ├── learnable_hyperbolic_adapter_albef_mscoco.pt
    ├── learnable_hyperbolic_adapter_mscoco.pt
    └── learnable_hyperbolic_adapter_clip_cnn_mscoco.pt
```

Before running:

1. Set `image_root` in `configs/Retrieval_flickr_re.yaml` and the three `configs/Hyperbolic_adapter*_coco.yaml` files.
2. Replace the machine-specific BERT, ALBEF, TCL, and CLIP paths in the selected launcher through its environment variables.
3. The current `eval_AET.py` resolves CLIP checkpoints from the local `prefix` inside `load_model`; change that directory to the folder containing `ViT-B-16.pt` and `RN101.pt`.
4. Keep the annotation-relative directory names shown above. For example, `flickr30k_test.json` expects `flickr30k-images/<image>.jpg` below `image_root`.

Do not commit third-party datasets or base-model checkpoints to this repository. Follow their original licenses and distribution terms.

## Hyperbolic adapter checkpoints

HySIA trains the adapter only on clean MSCOCO training pairs. The source VLM is frozen, validation images are disjoint from the training images, and no target-model gradient or feedback is used. At attack time the selected adapter is frozen.

The current best-checkpoint mapping is:

| Source model | Adapter checkpoint | Visual layers encoded by checkpoint | SHA-256 |
|---|---|---|---|
| ALBEF | `learnable_hyperbolic_adapter_albef_mscoco.pt` | `blocks.2,blocks.5,blocks.11` | `6f7b6f41b5524b2d16e5056d5fe2c4e07acfed9fa9c009113ef8f45108e01745` |
| TCL | reuse ALBEF adapter | `blocks.2,blocks.5,blocks.11` | same as above |
| CLIP-ViT | `learnable_hyperbolic_adapter_mscoco.pt` | `2,5,8,11` | `88c374352a0d3af40de25d8ac951f1605cf5d7aea2e9c2da325921e5004e98cf` |
| CLIP-CNN | `learnable_hyperbolic_adapter_clip_cnn_mscoco.pt` | `layer1,layer2,layer3,layer4`; grid 7 | `80ee86d057f4faebad23d460c109f93704700b2c5980c90dc1c0fdd5d196d344` |

Keep the matching `.json` metadata next to every released `.pt` file. The `.last.pt`, `_old.pt`, `_block11.pt`, and smoke checkpoints are training/debug artifacts rather than the default paper checkpoints.

> **Layer compatibility:** checkpoint layer names and `--hier_feature_layers` must match exactly. In particular, the current ALBEF best checkpoint contains three layers (`blocks.2,blocks.5,blocks.11`), even though the latest training config can be used to train a new four-layer adapter. Use the metadata of the checkpoint you actually load as the source of truth.

### Train adapters from scratch

After updating the paths in the corresponding YAML files:

```bash
# CLIP ViT-B/16
CUDA_ID=0 bash scripts/run_train_hyperbolic_adapter_coco.sh

# ALBEF
CUDA_ID=0 \
SOURCE_TEXT_ENCODER=bert-base-uncased \
ALBEF_CKPT=/path/to/mscoco.pth \
bash scripts/run_train_hyperbolic_adapter_albef_coco.sh

# CLIP RN101
CUDA_ID=0 \
SOURCE_TEXT_ENCODER=bert-base-uncased \
CLIP_MODEL_PATH=/path/to/RN101.pt \
bash scripts/run_train_hyperbolic_adapter_clip_cnn_coco.sh
```

Each run writes a best checkpoint, an optional last checkpoint, and JSON metadata under `checkpoints/hyperbolic_adapter/` unless `OUTPUT_DIR` is overridden.

## Flickr30K evaluation

The command below shows the paper-aligned ALBEF-source configuration. Update checkpoint paths first.

```bash
python -u eval_AET.py \
  --config configs/Retrieval_flickr_re.yaml \
  --seed 42 \
  --cuda_id 0 \
  --batch_size 4 \
  --source_model ALBEF \
  --source_text_encoder bert-base-uncased \
  --target_text_encoder bert-base-uncased \
  --albef_ckpt /path/to/albef_flickr30k.pth \
  --tcl_ckpt /path/to/tcl_flickr30k.pth \
  --original_rank_index_path std_eval_idx/flickr30k/ \
  --scales 0.5,0.75,1.25,1.5 \
  --attack_objective original_plus_hyperbolic \
  --lambda_hyp 0.2 \
  --adapter_checkpoint checkpoints/hyperbolic_adapter/learnable_hyperbolic_adapter_albef_mscoco.pt \
  --adapter_dim 256 \
  --adapter_train_data mscoco_train \
  --adapter_train_objective clean_ranking_logmeanexp \
  --adapter_input_normalization layernorm_l2 \
  --token_granularity patch \
  --hier_feature_layers blocks.2,blocks.5,blocks.11 \
  --layer_weight_strategy fixed_depth \
  --alignment_temperature 0.1 \
  --radius_temperature 0.1 \
  --ranking_temperature 0.1 \
  --hard_negative_topk 5 \
  --hyp_attack_loss_mode hybrid \
  --hyp_gradient_fusion_strategy candidate_guided \
  --hyp_geometry_mode hyperbolic \
  --hyperbolic_curvature 1.0 \
  --hyperbolic_eps 1e-5 \
  --radius_min 0.1 \
  --radius_max 0.8 \
  --attack_steps 10 \
  --image_eps 8 \
  --image_step_size 2 \
  --image_sample_numbers 5 \
  --text_num_perturbation 1 \
  --text_topk 10 \
  --text_threshold_pred_score 0.3 \
  --text_ratios 0.6,0.2,0.2 \
  --freeze_eval_models \
  --lazy_target_models \
  --result_file_path results/hysia_albef_flickr30k.txt
```

For another surrogate, change only the source-specific fields below while keeping the paper attack budget fixed:

| Source | `--source_model` | `--adapter_checkpoint` | `--hier_feature_layers` | Extra option |
|---|---|---|---|---|
| ALBEF | `ALBEF` | `learnable_hyperbolic_adapter_albef_mscoco.pt` | `blocks.2,blocks.5,blocks.11` | none |
| TCL | `TCL` | `learnable_hyperbolic_adapter_albef_mscoco.pt` | `blocks.2,blocks.5,blocks.11` | none |
| CLIP-ViT | `CLIP_ViT` | `learnable_hyperbolic_adapter_mscoco.pt` | `2,5,8,11` | none |
| CLIP-CNN | `CLIP_CNN` | `learnable_hyperbolic_adapter_clip_cnn_mscoco.pt` | `layer1,layer2,layer3,layer4` | `--cnn_token_grid_size 7` |

Convenience launchers are available under `scripts/`. Some retain older exploratory defaults, so override them with the paper values above when reporting paper-comparable results. In particular, use `LAMBDA_HYP=0.2`, `IMAGE_EPS=8`, `IMAGE_STEP_SIZE=2`, `HYP_ATTACK_LOSS_MODE=hybrid`, and `HYP_GRADIENT_FUSION_STRATEGY=candidate_guided`.

For a one-batch pipeline check, add:

```bash
--max_eval_batches 1
```

Smoke runs verify data/model/checkpoint compatibility but intentionally skip full ASR computation.

## Ablations and analysis

```bash
# Euclidean vs hyperbolic geometry and radial/ranking variants
MODE=full bash scripts/ablation/run_hysia_geometry_ablation_three_source.sh

# HEPM layer-selection variants
MODE=full bash scripts/ablation/run_hysia_hepm_layer_ablation_two_source.sh

# HITAM geometry and radial-discrepancy analysis
MODE=full bash scripts/ablation/run_hitam_geometry_ablation.sh
MODE=full bash scripts/ablation/run_hitam_radial_discrepancy.sh
```

Use `MODE=smoke` first to validate paths and GPU memory. Generated logs, raw result files, temporary manifests, and visualization intermediates are experiment artifacts and should not be committed by default.

## Tests

```bash
pytest -q \
  tests/test_hyperbolic_guidance.py \
  tests/test_vision_extractor_sequence_layout.py \
  tests/test_train_hyperbolic_adapter.py \
  tests/test_hitam_analysis.py
```

## Reproducibility notes

- Always record the Git commit, checkpoint SHA-256, dataset annotation hash, source model, selected visual layers, attack budget, and random seed.
- Keep source-specific adapters and layer layouts aligned. A mismatched adapter may load incorrectly or silently invalidate the intended hierarchy.
- The clean rank-index files in `std_eval_idx/flickr30k/` are required for the ASR calculation used by `eval_AET.py`.
- `--lazy_target_models` reduces peak GPU memory by moving target models on demand; it does not change the attack objective.
- The paper reports the protocol in the **Paper configuration** table. Several historical scripts and result files in this research workspace reflect earlier diagnostics and should not be treated as canonical paper settings.

## Citation

Please use the final publisher-provided citation when it becomes available. Until then, the local manuscript can be cited as:

```bibtex
@misc{jia2026hysia,
  title  = {HySIA: Hyperbolic Semantic Image--Text Association Method for Transferable Attacks against VLMs},
  author = {Jia, Yonggang and Wang, Kaiyu and Feng, Guangsheng and Lv, Hongwu and Gao, Sulin and Zheng, Wenqi},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgements

This implementation builds on the SA-AET attack pipeline and model components from ALBEF, TCL, and OpenAI CLIP. See the paper for the full bibliography and experimental discussion.

## License

See [LICENSE](LICENSE). Third-party datasets and checkpoints remain subject to their original licenses and terms of use.
