# HySIA: Hyperbolic Semantic Image–Text Association for Transferable Attacks against VLMs

Official PyTorch implementation of **HySIA**, a transferable multimodal adversarial attack that combines hierarchical visual supervision with structured image–text association in a Poincaré ball.

**Paper:** HySIA: Hyperbolic Semantic Image--Text Association Method for Transferable Attacks against VLMs  
**Authors:** Yonggang Jia, Kaiyu Wang, Guangsheng Feng, Hongwu Lv, Sulin Gao, and Wenqi Zheng  
**Institution:** College of Computer Science and Technology, Harbin Engineering University

## Abstract

Transferable adversarial examples expose vulnerabilities shared by different vision–language models (VLMs). Existing attacks usually optimize a flat Euclidean image–text objective using only final-layer visual representations. This can discard useful intermediate visual cues and make the attack overfit the surrogate model.

HySIA addresses these limitations with two components:

- **Hierarchical Visual Feature Extraction and Projection Module (HEPM):** extracts complementary patch-level representations from multiple visual-encoder depths and maps them, together with text representations, into a shared Poincaré ball.
- **Hyperbolic Image–Text Association Module (HITAM):** combines hyperbolic proximity and radial semantic-granularity discrepancy to construct structured token–caption associations for adversarial optimization.

The source encoder and the offline-trained hyperbolic adapter remain frozen during attack generation; only the image pixels are updated by projected gradient descent (PGD). Text perturbations are generated with BERT-Attack in the alternating image–text attack pipeline.

<p align="center">
  <img src="https://github.com/righteye7788/HySIA/blob/main/Jia-fig1v4.png" width="82%" alt="Motivation for HySIA">
</p>

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
├── HySIA.py                         # alternating text/image attack and objective fusion
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
