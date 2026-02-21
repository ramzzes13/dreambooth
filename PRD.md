# ModularBooth — Product Requirements Document (PRD)

**Version**: 1.0
**Date**: 2026-02-21
**Project**: Modular Multi-Subject Personalization for Diffusion Transformers with Disentangled Context-Appearance Learning

---

## 1. Executive Summary

ModularBooth is a research framework that extends DreamBooth for modern Diffusion Transformer (DiT) architectures (FLUX.1, SD3). It addresses three core limitations of the original DreamBooth: context-appearance entanglement, single-subject restriction, and full model fine-tuning overhead.

The framework consists of three stages:
1. **Blockwise Subject Encoding** — Adaptive LoRA parameterization localized to identity-critical DiT blocks
2. **Contrastive Context Disentanglement (CCD)** — Contrastive loss separating subject identity from background context
3. **Masked Compositional Inference** — Token-aware cross-attention masking for multi-subject generation

---

## 2. Architecture Overview

```
modularbooth/
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Default hyperparameters
│   ├── flux.yaml               # FLUX.1-dev specific config
│   └── sd3.yaml                # Stable Diffusion 3 config
├── data/                       # Data pipeline
│   ├── __init__.py
│   ├── dataset.py              # DreamBooth dataset with augmentation
│   ├── augmentation.py         # Background replacement, color jitter, style transfer
│   ├── captioning.py           # LLM-based informative captioning
│   └── benchmark.py            # Benchmark dataset construction
├── models/                     # Model components
│   ├── __init__.py
│   ├── blockwise_lora.py       # Blockwise LoRA parameterization for DiT
│   ├── knowledge_probe.py      # DiT block knowledge probing
│   ├── attention_mask.py       # Token-aware cross-attention masking
│   └── lora_merge.py           # Multi-subject LoRA composition
├── losses/                     # Loss functions
│   ├── __init__.py
│   ├── ccd_loss.py             # Contrastive Context Disentanglement loss
│   ├── prior_preservation.py   # Prior Preservation Loss (PPL)
│   └── combined.py             # Combined training objective
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── trainer.py              # Main training loop
│   ├── scheduler.py            # LR scheduling and warmup
│   └── callbacks.py            # Logging, checkpointing, validation callbacks
├── evaluation/                 # Evaluation metrics
│   ├── __init__.py
│   ├── dino_score.py           # DINO / DINOv2 subject fidelity
│   ├── clip_score.py           # CLIP-I and CLIP-T scores
│   ├── identity_isolation.py   # Identity Isolation Score (IIS)
│   ├── entanglement.py         # Context-Appearance Entanglement (CAE)
│   ├── diversity.py            # LPIPS diversity metric
│   ├── vqa_alignment.py        # VQA-based prompt alignment
│   └── run_evaluation.py       # Full evaluation pipeline
├── inference/                  # Inference pipeline
│   ├── __init__.py
│   ├── single_subject.py       # Single-subject generation
│   ├── multi_subject.py        # Multi-subject masked composition
│   └── layout.py               # Layout generation and bounding box utils
├── scripts/                    # CLI scripts
│   ├── train.py                # Train a subject LoRA module
│   ├── evaluate.py             # Run evaluation metrics
│   ├── generate.py             # Generate images (single or multi-subject)
│   ├── probe_blocks.py         # Run knowledge probing on DiT blocks
│   └── run_ablation.py         # Run ablation studies
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_blockwise_lora.py  # LoRA module tests
│   ├── test_ccd_loss.py        # CCD loss computation tests
│   ├── test_attention_mask.py  # Attention masking tests
│   ├── test_metrics.py         # Evaluation metric tests
│   ├── test_dataset.py         # Dataset pipeline tests
│   ├── test_training.py        # Training loop smoke tests
│   └── test_inference.py       # Inference pipeline tests
├── pyproject.toml              # Project metadata and dependencies
└── README.md                   # (not created — exists in research plan)
```

---

## 3. Atomic Tasks

### Phase 0: Project Foundation

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P0-1 | Create `pyproject.toml` | Define project metadata, dependencies (torch, diffusers, transformers, peft, accelerate, safetensors, etc.) | `pip install -e .` succeeds; all imports resolve |
| P0-2 | Create `configs/default.yaml` | Default hyperparameters for all training, evaluation, inference settings | YAML loads without error; all expected keys present |
| P0-3 | Create `configs/flux.yaml` | FLUX.1-dev specific overrides (model ID, block structure, attention heads) | Config merges correctly with defaults |
| P0-4 | Create `configs/sd3.yaml` | SD3 Medium specific overrides | Config merges correctly with defaults |

### Phase 1: Data Pipeline

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P1-1 | Implement `data/dataset.py` | `DreamBoothDataset` class: loads subject images, pairs with captions, returns training-ready batches | Unit test: creates dataset from dummy images, iterates batches, correct tensor shapes |
| P1-2 | Implement `data/augmentation.py` | Background augmentation pipeline: SAM-2 segmentation → background replacement via inpainting, color jitter, style transfer | Unit test: given an image + mask, produces K augmented variants with subject intact |
| P1-3 | Implement `data/captioning.py` | LLM-based captioning: generates informative descriptions with rare-token identifiers | Unit test: given image path, returns caption string containing `[V]` token |
| P1-4 | Implement `data/benchmark.py` | Benchmark dataset construction: loads DreamBooth benchmark, constructs multi-subject prompt sets | Unit test: returns correct number of subjects, prompts, and complexity tiers |

### Phase 2: Model Components

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P2-1 | Implement `models/blockwise_lora.py` | `BlockwiseLoRA` class: applies adaptive-rank LoRA to DiT blocks based on probing results. Supports FLUX and SD3 block structures. | Unit test: applies LoRA to dummy DiT, verifies parameter counts per block match config, forward pass produces correct shape |
| P2-2 | Implement `models/knowledge_probe.py` | `KnowledgeProbe` class: trains single-block LoRA, measures DINO/CLIP per block, outputs block classification (identity/context/shared) | Unit test: probing protocol runs on mock model, produces per-block scores |
| P2-3 | Implement `models/attention_mask.py` | `TokenAwareAttentionMask`: computes spatial masks from bounding boxes, applies soft masking to cross-attention, implements negative attention | Unit test: given bounding boxes, produces correct mask shapes; negative attention suppresses outside regions |
| P2-4 | Implement `models/lora_merge.py` | `LoRAComposer`: loads N subject LoRA modules, applies weighted merging within spatial regions | Unit test: loads 2 dummy LoRA checkpoints, merges, verifies output shape and parameter blending |

### Phase 3: Loss Functions

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P3-1 | Implement `losses/ccd_loss.py` | `CCDLoss`: contrastive loss on intermediate DiT features pooled over subject regions. Positive pairs = same subject different context, negatives = different subjects or background. | Unit test: given mock features and masks, loss computes to finite positive value; loss decreases when positive pairs are more similar |
| P3-2 | Implement `losses/prior_preservation.py` | `PriorPreservationLoss`: standard PPL from DreamBooth — diffusion loss on class-generated samples | Unit test: given model predictions and targets, loss matches expected MSE computation |
| P3-3 | Implement `losses/combined.py` | `ModularBoothLoss`: combines diffusion loss + PPL + CCD with configurable weights λ1, λ2 | Unit test: combined loss = sum of weighted components; gradients flow through all terms |

### Phase 4: Training Pipeline

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P4-1 | Implement `training/trainer.py` | `ModularBoothTrainer`: main training loop with AdamW, gradient accumulation, mixed precision, logging | Smoke test: trains 5 steps on dummy data without error; loss decreases |
| P4-2 | Implement `training/scheduler.py` | LR scheduler with linear warmup and cosine decay | Unit test: LR values match expected schedule at key steps |
| P4-3 | Implement `training/callbacks.py` | `CheckpointCallback`, `LoggingCallback`, `ValidationCallback` | Unit test: checkpoint saves/loads correctly; logging produces expected format |

### Phase 5: Evaluation Metrics

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P5-1 | Implement `evaluation/dino_score.py` | DINO and DINOv2 subject fidelity: pairwise cosine similarity between generated and reference image embeddings | Unit test: identical images → score ~1.0; random images → score ~0.0-0.3 |
| P5-2 | Implement `evaluation/clip_score.py` | CLIP-I (image-image) and CLIP-T (text-image) scores | Unit test: matching text-image pair scores higher than mismatched |
| P5-3 | Implement `evaluation/identity_isolation.py` | IIS: for two-subject scenes, crop regions → DINO similarity to correct vs. incorrect reference | Unit test: correctly-matched crops produce positive IIS |
| P5-4 | Implement `evaluation/entanglement.py` | CAE: variance of DINO embeddings across contexts for same subject. Lower = less entanglement | Unit test: identical crops across contexts → CAE ~0; varied crops → higher CAE |
| P5-5 | Implement `evaluation/diversity.py` | LPIPS diversity between generated images for same prompt | Unit test: identical images → LPIPS ~0; different images → LPIPS > 0 |
| P5-6 | Implement `evaluation/vqa_alignment.py` | VQA-based prompt alignment using a VLM | Unit test: returns accuracy float between 0 and 1 |
| P5-7 | Implement `evaluation/run_evaluation.py` | Full evaluation pipeline: loads generated images, runs all metrics, outputs JSON report | Integration test: runs on dummy data, produces valid JSON with all metric keys |

### Phase 6: Inference Pipeline

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P6-1 | Implement `inference/single_subject.py` | Single-subject generation: loads LoRA, applies to DiT, generates images | Smoke test: runs with mock pipeline, produces tensor of correct shape |
| P6-2 | Implement `inference/multi_subject.py` | Multi-subject composition: loads N LoRAs, applies spatial masking, generates | Smoke test: runs with 2 mock LoRAs and bounding boxes |
| P6-3 | Implement `inference/layout.py` | Layout utilities: bounding box parsing, LLM layout generation, spatial arrangement | Unit test: generates non-overlapping bounding boxes for N subjects |

### Phase 7: CLI Scripts

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P7-1 | Implement `scripts/train.py` | CLI entry point for training with argparse/hydra config | `python scripts/train.py --help` runs without error |
| P7-2 | Implement `scripts/evaluate.py` | CLI entry point for evaluation | `python scripts/evaluate.py --help` runs without error |
| P7-3 | Implement `scripts/generate.py` | CLI entry point for image generation | `python scripts/generate.py --help` runs without error |
| P7-4 | Implement `scripts/probe_blocks.py` | CLI entry point for knowledge probing | `python scripts/probe_blocks.py --help` runs without error |
| P7-5 | Implement `scripts/run_ablation.py` | CLI entry point for ablation studies | `python scripts/run_ablation.py --help` runs without error |

### Phase 8: Tests

| ID | Task | Description | Verification |
|----|------|-------------|--------------|
| P8-1 | Write `tests/test_blockwise_lora.py` | Unit tests for blockwise LoRA creation, forward pass, save/load | All tests pass |
| P8-2 | Write `tests/test_ccd_loss.py` | Unit tests for CCD loss computation, gradient flow | All tests pass |
| P8-3 | Write `tests/test_attention_mask.py` | Unit tests for spatial masking, negative attention | All tests pass |
| P8-4 | Write `tests/test_metrics.py` | Unit tests for all evaluation metrics | All tests pass |
| P8-5 | Write `tests/test_dataset.py` | Unit tests for dataset loading and augmentation | All tests pass |
| P8-6 | Write `tests/test_training.py` | Smoke tests for training loop | All tests pass |
| P8-7 | Write `tests/test_inference.py` | Smoke tests for inference pipeline | All tests pass |

---

## 4. Verification Strategy

### 4.1 Unit-Level Verification

Every module has targeted unit tests that verify:
- **Correctness**: Outputs match expected values for known inputs
- **Shape consistency**: Tensor shapes propagate correctly through the pipeline
- **Gradient flow**: All trainable parameters receive gradients
- **Numerical stability**: No NaN/Inf values under normal operation
- **Edge cases**: Empty batches, single-image subjects, overlapping bounding boxes

### 4.2 Integration-Level Verification

| Integration Test | What It Verifies |
|-----------------|-----------------|
| Dataset → Training | Data pipeline produces correctly shaped batches that the trainer consumes |
| Training → Checkpoint → Inference | Saved LoRA weights load correctly and produce valid images |
| Multi-LoRA → Masked Inference | Multiple independently trained LoRAs compose without crashing |
| Inference → Evaluation | Generated images pass through all evaluation metrics without error |
| Config → All Components | YAML configs correctly parameterize every module |

### 4.3 Numerical Verification

| Check | Method |
|-------|--------|
| CCD loss convergence | Positive pairs should produce lower loss than negative pairs; verify with synthetic data where ground truth is known |
| PPL behavior | Prior preservation loss should remain stable (not decrease to zero, not explode); monitor via callback |
| DINO score sanity | Score should be ~1.0 for identical images, ~0.5-0.7 for same-class different-instance, <0.3 for different classes |
| CLIP-T sanity | Score should be >0.25 for matching prompt-image, <0.20 for mismatched |
| IIS sign | Should be positive when subjects are correctly assigned to regions |
| CAE monotonicity | Should decrease as CCD loss weight increases (ablation verification) |
| LoRA parameter count | Should match `rank × (d_in + d_out) × num_blocks_with_lora`; verify against config |

### 4.4 Smoke Tests (No GPU Required)

All smoke tests run on CPU with:
- Tiny model stubs (2-layer DiT with small hidden dim)
- Synthetic random data (no real images needed)
- 1-5 training steps
- Purpose: verify the pipeline runs end-to-end without errors

### 4.5 Research Verification Checklist

When actual training is run (on GPU), verify:

- [ ] Single-subject DINO score ≥ 0.65 (baseline: DreamBooth-LoRA ~0.66)
- [ ] Single-subject CLIP-T ≥ 0.30 (baseline: DreamBooth ~0.305)
- [ ] Multi-subject IIS > 0.30 (baseline: TARA ~0.35)
- [ ] CAE reduction ≥ 15% vs. no-CCD baseline
- [ ] Training completes in <10 min per subject on A100
- [ ] LoRA checkpoint size < 10 MB per subject
- [ ] Inference produces 1024×1024 images in <15 seconds on RTX 4090
- [ ] All ablation conditions produce distinct, explainable results

---

## 5. Dependencies

### Core
- `torch>=2.1.0` — PyTorch with CUDA support
- `diffusers>=0.30.0` — HuggingFace Diffusers (FLUX, SD3 support)
- `transformers>=4.40.0` — Tokenizers, CLIP, vision models
- `peft>=0.12.0` — Parameter-Efficient Fine-Tuning (LoRA)
- `accelerate>=0.30.0` — Mixed precision, distributed training
- `safetensors>=0.4.0` — Safe checkpoint serialization

### Evaluation
- `torchvision>=0.16.0` — Image transforms
- `torchmetrics>=1.3.0` — LPIPS and other metrics
- `open-clip-torch>=2.24.0` — OpenCLIP for CLIP scores
- `lpips>=0.1.4` — Learned Perceptual Image Patch Similarity

### Data
- `Pillow>=10.0.0` — Image loading/manipulation
- `segment-anything-2` — SAM-2 for subject segmentation
- `albumentations>=1.3.0` — Image augmentation

### Infrastructure
- `omegaconf>=2.3.0` — YAML config management
- `wandb>=0.16.0` — Experiment tracking
- `tqdm>=4.66.0` — Progress bars
- `pytest>=8.0.0` — Testing framework

---

## 6. Configuration Schema

```yaml
# default.yaml
model:
  backbone: "black-forest-labs/FLUX.1-dev"
  dtype: "bfloat16"

lora:
  identity_rank: 16
  context_rank: 4
  shared_rank: 8
  alpha_ratio: 1.0  # alpha = rank
  dropout: 0.05
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]

training:
  num_steps: 800
  batch_size: 1
  gradient_accumulation: 4
  learning_rate: 1.0e-4
  optimizer: "adamw"
  weight_decay: 0.01
  warmup_steps: 50
  mixed_precision: "bf16"
  seed: 42

prior_preservation:
  enabled: true
  num_class_images: 200
  lambda_ppl: 1.0

ccd:
  enabled: true
  lambda_ccd: 0.3
  temperature: 0.07
  num_augmentations: 5

inference:
  num_steps: 30
  guidance_scale: 7.5
  negative_attention_strength: 3.0
  mask_leakage_alpha: 0.05
  resolution: 1024

evaluation:
  dino_model: "facebook/dino-vits16"
  dinov2_model: "facebook/dinov2-vitb14"
  clip_model: "openai/clip-vit-large-patch14"
  num_images_per_prompt: 4
```

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| FLUX.1-dev API/weights access issues | Blocks all development | Support SD3 as fallback; abstract backbone behind interface |
| SAM-2 segmentation failures on edge cases | CCD loss trains on noisy masks | Mask quality filtering; confidence thresholding |
| DiT block probing shows no clean identity/context split | Undermines blockwise strategy | Graceful fallback to uniform LoRA with soft rank weighting |
| CCD loss causes training instability | Model diverges | Gradient clipping; CCD warmup (enable after N steps); reduce λ2 |
| Multi-subject masking degrades image quality | Poor composition results | Tune α carefully; add soft feathering at mask boundaries |
| Evaluation metrics disagree (high DINO, low CLIP-T) | Unclear results | Weighted composite score; user study as ground truth |
