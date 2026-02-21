# Research Plan: Modular Multi-Subject Personalization for Diffusion Transformers with Disentangled Context-Appearance Learning

---

## 1. Introduction

### 1.1 Problem Statement

Personalized text-to-image generation—the ability to synthesize photorealistic images of a specific subject in novel contexts given only a handful of reference images—remains a central challenge in generative AI. Users want to place *their* dog in the Grand Canyon, *their* teapot in a Vermeer painting, or *their* backpack on the surface of Mars. While large-scale text-to-image models possess extraordinary generative diversity, they cannot faithfully reproduce the fine-grained visual identity of a particular subject from text alone.

This problem is practically important for creative professionals (personalized marketing assets, concept art), consumer applications (AI photo booths, social media content), and scientific visualization (generating controlled stimuli for perception studies). The challenge intensifies when users wish to compose *multiple* personalized subjects in a single scene—a scenario that remains largely unsolved.

### 1.2 Summary of the Original Paper

Ruiz et al. (2023) introduced **DreamBooth**, a landmark method for subject-driven generation. Given 3–5 casually captured images of a subject, DreamBooth fine-tunes all layers of a pre-trained text-to-image diffusion model (Imagen or Stable Diffusion) so that the subject is bound to a rare-token identifier (e.g., "a [V] dog"). The method's key contributions include:

1. **Rare-token identifier design**: Selecting tokens with weak priors in the vocabulary to minimize interference with existing semantics.
2. **Class-specific prior preservation loss (PPL)**: Supervising the fine-tuned model with its own pre-generated class samples to combat language drift and maintain output diversity.
3. **A new dataset and evaluation protocol**: 30 subjects, 25 prompts, and metrics including DINO-based subject fidelity (DINO), CLIP-based image similarity (CLIP-I), and prompt fidelity (CLIP-T).

DreamBooth significantly outperformed the concurrent Textual Inversion approach on both subject fidelity (DINO: 0.696 vs. 0.569) and prompt fidelity (CLIP-T: 0.306 vs. 0.255), with 68% user preference for subject fidelity and 81% for prompt fidelity.

### 1.3 Limitations of DreamBooth

Despite its impact, DreamBooth exhibits several limitations acknowledged by the authors and identified by subsequent work:

1. **Context-appearance entanglement**: The subject's appearance can change due to the prompted context (e.g., a backpack changing color when placed on blue fabric).
2. **Overfitting to reference images**: When prompts resemble the original training environment, generated images collapse toward the reference set.
3. **Single-subject restriction**: DreamBooth personalizes one subject per fine-tuned model. Composing multiple personalized subjects requires training separate models and has no principled merging strategy.
4. **Full model fine-tuning overhead**: Each subject requires fine-tuning all model parameters (~1–2 GB per checkpoint), making storage and deployment impractical at scale.
5. **UNet-centric architecture**: DreamBooth was designed for UNet-based diffusion models (Imagen, Stable Diffusion 1.x/2.x). Modern Diffusion Transformers (DiT) such as FLUX, Stable Diffusion 3, and Sora use fundamentally different architectures with MM-DiT blocks.
6. **Limited evaluation scope**: The benchmark covers 30 subjects with relatively simple prompts; it does not assess multi-subject composition, temporal consistency (video), or 3D generation.

### 1.4 Goal of the Proposed Work

This thesis proposes **ModularBooth**, a framework for *modular, multi-subject personalization* on modern Diffusion Transformer (DiT) architectures that explicitly disentangles context from appearance during fine-tuning. The key contributions are:

1. A **blockwise-parameterized LoRA** adaptation strategy tailored to DiT architectures that localizes subject knowledge in specific transformer blocks, enabling plug-and-play composition.
2. A **contrastive disentanglement loss** that decorrelates subject appearance from background context during training, directly addressing context-appearance entanglement.
3. A **cross-attention masking inference protocol** that prevents identity leakage when composing multiple independently trained subject modules.
4. Comprehensive evaluation on an **extended multi-subject benchmark** that goes beyond DreamBooth's original 30-subject single-subject setup to include 2- and 3-subject composition scenarios, prompt complexity tiers, and a new entanglement metric.

---

## 2. Literature Review

### 2.1 Foundational Work: DreamBooth and Textual Inversion

**DreamBooth** (Ruiz et al., 2023) established the paradigm of fine-tuning all diffusion model parameters for subject-driven generation, using rare-token identifiers and a prior preservation loss to maintain class priors during few-shot adaptation. **Textual Inversion** (Gal et al., 2022) proposed an alternative strategy: freezing the diffusion model entirely and optimizing only a new token embedding to represent the subject. While more parameter-efficient, Textual Inversion is limited by the expressiveness of the frozen model and achieves substantially lower fidelity.

### 2.2 Parameter-Efficient Personalization

The tension between DreamBooth's quality and Textual Inversion's efficiency has driven extensive research into parameter-efficient alternatives:

- **Custom Diffusion** (Kumari et al., 2023) fine-tunes only the cross-attention key and value projection matrices, reducing trainable parameters by ~75% while supporting multi-concept composition through constrained optimization.
- **LoRA for Diffusion** (Hu et al., 2021; adapted by community): Low-Rank Adaptation decomposes weight updates into low-rank matrices, reducing storage to ~3–4 MB per subject. However, naively merging multiple LoRA modules causes attribute interference.
- **BlockLoRA** (2025) introduces blockwise LoRA parameterization with Randomized Output Erasure, enabling merging of up to 15 concepts with minimal interference.
- **TARA (Token-Aware LoRA)** (2025) assigns token masks to individual LoRA modules, constraining each module to attend only to its associated rare token during multi-concept inference, preventing spatial interference.

### 2.3 Addressing Overfitting and Entanglement

DreamBooth's core failure modes have received targeted attention:

- **SID (Selectively Informative Description)** (2024) uses multimodal LLMs (GPT-4V) to generate rich text descriptions of reference images, reducing embedding entanglement by providing the optimizer with contextual information that disambiguates subject from background.
- **MINDiff (Mask-Integrated Negative Attention)** (2025) introduces negative attention at inference time to suppress subject influence in irrelevant spatial regions. Unlike PPL, it requires no retraining and can be applied post-hoc to any DreamBooth model.
- **DreamBlend** (2025) addresses the fidelity-diversity trade-off by blending early-checkpoint features (high prompt fidelity) with late-checkpoint features (high subject fidelity) during cross-attention-guided inference.
- **Lipschitz-bounded training** (2025) constrains distributional drift during fine-tuning with a Lipschitz objective, preventing catastrophic forgetting in data-scarce scenarios.

### 2.4 Encoder-Based (Zero-Shot) Personalization

A parallel line of work avoids per-subject fine-tuning entirely:

- **IP-Adapter** (Ye et al., 2023) injects CLIP image features via decoupled cross-attention, enabling zero-shot subject conditioning. However, it struggles with fine-grained identity preservation.
- **Conceptrol** (2025) improves IP-Adapter by introducing a textual concept mask that constrains visual attention, achieving 89% improvement over vanilla IP-Adapter on personalization benchmarks.
- **Personalize Anything** (2025) exploits DiT's internal representations via timestep-adaptive token replacement, achieving zero-shot subject reconstruction without any training or adapter modules.
- **FreeCus** (ICCV 2025) provides training-free subject customization for FLUX-series DiTs using attention sharing and MLLM-enriched semantic representations.

### 2.5 Preference-Aligned Personalization

Recent work integrates human preference alignment into personalization:

- **DreamBoothDPO** (2025) applies Direct Preference Optimization to DreamBooth, using synthetic paired datasets scored by external quality metrics to balance concept fidelity and contextual alignment without human annotation.
- **PPD (Personalized Preference Fine-tuning)** (2025) extracts personal preference embeddings from pairwise comparisons using a VLM, achieving 76% win rate over Stable Cascade in generating user-aligned images.

### 2.6 Multi-Subject Composition

Composing multiple personalized subjects in a single image is an open challenge:

- **MuDI** (2024) uses identity decoupling via segmented data augmentation, achieving 2× the success rate of baselines in preventing identity mixing between two subjects.
- **LoRACLR** (CVPR 2025) merges multiple subject LoRA modules using contrastive weight-space alignment, preventing attribute leakage during multi-concept generation.
- **TokenVerse** (2025) disentangles visual elements from a single image via DiT modulation-space (shift/scale) manipulation, enabling plug-and-play recombination of concepts.
- **LayerComposer** (2025) uses a layered canvas representation where each subject occupies a distinct layer with transparent latent pruning for scalable multi-human generation.

### 2.7 Extensions to Video and 3D

Subject-driven generation has expanded beyond static 2D images:

- **VideoBooth** (2024) extends DreamBooth principles to video generation using image prompts with diffusion-based video models.
- **Video Alchemist** (2025) enables multi-subject, open-set personalization in video generation via a DiT module that fuses reference images with subject-level text prompts.
- **CP-GS** (2025) enables personalized 3D scene generation from a single image by progressively propagating reference appearance to novel views via iterative LoRA fine-tuning and Gaussian Splatting.

### 2.8 Summary and Gap Analysis

Despite significant progress, no existing method simultaneously achieves:
1. High-fidelity single- and multi-subject personalization on modern DiT architectures.
2. Explicit disentanglement of subject identity from contextual appearance during training.
3. Modular, plug-and-play composition of independently trained subject modules without retraining.
4. Practical storage and compute requirements (single GPU, minutes of training, megabytes of storage).

Our proposed **ModularBooth** framework targets this gap.

---

## 3. Proposed Methodology

### 3.1 Overview

ModularBooth is a three-stage framework:

**Stage 1 — Blockwise Subject Encoding**: For each subject, train a lightweight, blockwise-parameterized LoRA module on a DiT-based backbone (FLUX.1 or Stable Diffusion 3), localizing subject knowledge in specific transformer blocks identified via knowledge probing.

**Stage 2 — Contrastive Context Disentanglement**: During LoRA training, apply a contrastive loss that pushes apart the latent representations of the subject-in-original-context from subject-in-augmented-context, while pulling together representations of the same subject identity across diverse backgrounds.

**Stage 3 — Masked Compositional Inference**: At inference time, compose multiple subject LoRA modules using token-aware cross-attention masking and spatial layout guidance, preventing identity leakage between subjects.

### 3.2 Stage 1: Blockwise Subject Encoding for DiTs

#### 3.2.1 Knowledge Localization in DiTs

Recent work (Zarei et al., 2025) has shown that different DiT blocks encode qualitatively different types of knowledge: early blocks capture layout and structure, middle blocks encode semantic content, and late blocks refine texture and fine-grained details. We leverage this insight by *probing* a pretrained DiT to identify which blocks are most critical for subject identity versus contextual appearance.

**Probing protocol**: For a set of 50 diverse subjects, we fine-tune single-block LoRA modules and measure DINO subject fidelity and CLIP-T prompt fidelity per block. Blocks with high subject fidelity and low prompt fidelity degradation are designated as "identity blocks"; blocks with the opposite profile are designated as "context blocks."

#### 3.2.2 Blockwise LoRA Parameterization

Based on the probing results, we assign LoRA rank adaptively:
- **Identity blocks**: Higher LoRA rank (r = 16–32) to capture fine-grained subject details.
- **Context blocks**: No LoRA or minimal rank (r = 1–4) to preserve the model's contextual generation prior.
- **Shared blocks**: Moderate rank (r = 8) with regularization.

This results in a compact, modular subject representation (~2–5 MB per subject) that encodes subject identity without overwriting contextual knowledge.

#### 3.2.3 Training Procedure

For each subject (3–5 reference images):

1. Caption each reference image using a multimodal LLM (e.g., LLaVA or GPT-4V) to generate selectively informative descriptions that include both the rare-token identifier and contextual details. This follows the SID insight that richer captions reduce entanglement.
2. Fine-tune the blockwise LoRA parameters using the standard diffusion denoising objective with the DreamBooth prompt format ("a [V] [class noun]") combined with the informative captions.
3. Apply the prior preservation loss with class-generated samples, following DreamBooth's original formulation.

Training takes approximately 500–1000 steps (3–5 minutes on a single A100 GPU).

### 3.3 Stage 2: Contrastive Context Disentanglement

#### 3.3.1 Motivation

DreamBooth's prior preservation loss addresses language drift but does not explicitly disentangle subject appearance from the context in reference images. If all reference images show a red backpack against a wooden table, the model may associate "red" or "wooden texture" with the subject token.

#### 3.3.2 Contrastive Loss Formulation

We propose a **Contextual Contrastive Disentanglement (CCD) loss** that operates on intermediate DiT features during training:

**Data augmentation**: For each reference image, generate $K$ context-augmented variants using:
- Background replacement via inpainting (using a segmentation mask of the subject).
- Color jittering of the background only.
- Style transfer applied to the background only.

This yields pairs: (original image, augmented image) where the subject is identical but the context differs.

**Contrastive objective**: Let $\mathbf{h}_i^s$ denote the intermediate DiT features at the identity blocks for the $i$-th image, pooled over the spatial region corresponding to the subject (obtained via automatic segmentation). The CCD loss is:

$$
\mathcal{L}_{\text{CCD}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i^s, \mathbf{h}_j^s) / \tau)}{\sum_{k=1}^{2K} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{h}_i^s, \mathbf{h}_k^s) / \tau)}
$$

where $(\mathbf{h}_i^s, \mathbf{h}_j^s)$ are positive pairs (same subject, different context), negatives are features from different subjects or the background regions of the same image, $\text{sim}(\cdot, \cdot)$ is cosine similarity, and $\tau$ is a temperature parameter.

#### 3.3.3 Total Training Loss

The complete training objective combines three terms:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \mathcal{L}_{\text{PPL}} + \lambda_2 \mathcal{L}_{\text{CCD}}
$$

where $\mathcal{L}_{\text{diffusion}}$ is the standard denoising loss (Eq. 1 in DreamBooth), $\mathcal{L}_{\text{PPL}}$ is the prior preservation loss (Eq. 2 in DreamBooth), and $\lambda_1, \lambda_2$ are weighting coefficients. We anticipate $\lambda_1 = 1.0$ (following DreamBooth) and $\lambda_2 \in [0.1, 0.5]$ (tuned via ablation).

### 3.4 Stage 3: Masked Compositional Inference

#### 3.4.1 Multi-Subject Composition Protocol

Given $N$ independently trained subject LoRA modules $\{\Delta\theta_1, \ldots, \Delta\theta_N\}$ and a composite text prompt ("a [V1] dog playing with a [V2] cat on a beach"):

1. **Layout generation**: Use an LLM (or user-provided layout) to assign each subject a bounding box in the target image.
2. **Token-aware attention masking**: During the cross-attention computation in the DiT, restrict each subject LoRA's influence to spatial tokens within its assigned bounding box. Formally, for subject $n$ with bounding box $B_n$, we define a soft spatial mask:

$$
M_n(x, y) = \begin{cases} 1 & \text{if } (x, y) \in B_n \\ \alpha & \text{otherwise} \end{cases}
$$

where $\alpha \ll 1$ (e.g., 0.05) allows minimal global influence for scene coherence.

3. **Weighted LoRA merging**: Within each spatial region, apply only the corresponding subject's LoRA weights. In overlapping regions, blend LoRA contributions using distance-based soft weighting.

#### 3.4.2 Negative Attention for Identity Isolation

Inspired by MINDiff (2025), we optionally apply negative cross-attention for each subject's token outside its designated region, actively suppressing identity leakage:

$$
\text{Attention}_n = \text{softmax}\left(\frac{QK_n^T}{\sqrt{d}} - \gamma \cdot (1 - M_n)\right) V_n
$$

where $\gamma > 0$ controls the suppression strength.

### 3.5 Textual Description of System Architecture

The overall ModularBooth pipeline can be visualized as follows:

```
Input: 3-5 images of Subject A, 3-5 images of Subject B
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   [Stage 1+2: Train               [Stage 1+2: Train
    Blockwise LoRA A                Blockwise LoRA B
    + CCD Loss]                     + CCD Loss]
          │                               │
          ▼                               ▼
   LoRA Module A                   LoRA Module B
   (~3 MB)                         (~3 MB)
          │                               │
          └───────────┬───────────────────┘
                      ▼
              [Stage 3: Masked
               Compositional
               Inference]
                      │
                      ▼
            Composite Image:
     "a [V1] dog playing with
      a [V2] cat on a beach"
```

---

## 4. Experimental Design

### 4.1 Datasets

#### 4.1.1 DreamBooth Benchmark (Original)
- **30 subjects** (21 objects, 9 live subjects/pets), 3–5 images each.
- **25 prompts** per category (recontextualization, property modification, accessorization).
- 4 images generated per subject-prompt pair → 3,000 images total.
- Purpose: Direct comparison with DreamBooth and subsequent baselines.

#### 4.1.2 Extended Multi-Subject Benchmark (New)
We will construct a new benchmark specifically for multi-subject composition:
- **50 subjects** (30 objects, 20 animals/people), sourced from DreamBooth dataset + Unsplash + custom captures.
- **Multi-subject prompts**: 30 two-subject prompts and 15 three-subject prompts describing interactions ("a [V1] dog sitting next to a [V2] vase on a shelf").
- **Entanglement-probing prompts**: 20 prompts designed to trigger context-appearance entanglement (e.g., placing a red object on a blue surface).
- **Complexity tiers**: Simple (single background), medium (subject interacts with environment), hard (multiple subjects interact with each other and environment).
- 4 images per prompt → ~2,000 images for multi-subject evaluation.

#### 4.1.3 Context-Augmented Training Data
For each subject, we generate 5 background-augmented variants per reference image using SAM-2 segmentation + inpainting, yielding 15–25 augmented training images to support the CCD loss.

### 4.2 Baselines

We will compare ModularBooth against the following methods:

| Method | Type | Multi-Subject | Architecture |
|--------|------|---------------|--------------|
| DreamBooth (Ruiz et al., 2023) | Full fine-tuning | No | UNet (SD 1.5) |
| Textual Inversion (Gal et al., 2022) | Embedding only | No | UNet (SD 1.5) |
| Custom Diffusion (Kumari et al., 2023) | Cross-attn fine-tuning | Yes (joint) | UNet (SD 1.5) |
| DreamBooth-LoRA | LoRA fine-tuning | Naive merge | UNet (SD 1.5) |
| TARA (2025) | Token-aware LoRA | Yes (modular) | UNet (SDXL) |
| LoRACLR (CVPR 2025) | Contrastive LoRA merging | Yes (merged) | UNet (SDXL) |
| IP-Adapter (Ye et al., 2023) | Encoder-based | Yes (zero-shot) | UNet (SDXL) |
| FreeCus (ICCV 2025) | Training-free DiT | Yes (zero-shot) | DiT (FLUX) |
| Personalize Anything (2025) | Training-free DiT | Yes (zero-shot) | DiT (FLUX) |
| **ModularBooth (Ours)** | Blockwise LoRA + CCD | Yes (modular) | DiT (FLUX) |

### 4.3 Evaluation Metrics

#### 4.3.1 Subject Fidelity
- **DINO score**: Average pairwise cosine similarity between ViT-S/16 DINO embeddings of generated and real subject images. (Primary metric, following DreamBooth.)
- **CLIP-I score**: Average pairwise cosine similarity between CLIP ViT-L/14 embeddings of generated and real images.
- **DINOv2 score**: Updated metric using DINOv2 ViT-B/14 for stronger fine-grained distinction (new addition).

#### 4.3.2 Prompt Fidelity
- **CLIP-T score**: Average cosine similarity between CLIP text and image embeddings for the prompt and generated image.
- **VQA-based prompt alignment**: Use a VQA model (e.g., LLaVA-1.5) to answer binary questions about prompt elements present in the generated image ("Is there a beach in this image?"). Report accuracy.

#### 4.3.3 Multi-Subject Metrics (New)
- **Identity Isolation Score (IIS)**: For two-subject scenes, crop each subject region and compute DINO similarity to the correct reference subject minus DINO similarity to the other subject. Higher = less identity mixing.
- **Composition Accuracy**: Fraction of generated images where a VQA model confirms the presence and approximate spatial arrangement of all requested subjects.

#### 4.3.4 Entanglement Metric (New)
- **Context-Appearance Entanglement (CAE)**: Generate the same subject in 10 different contexts. Extract subject crops. Compute the variance of DINO embeddings across contexts. Lower variance = less entanglement (the subject appearance remains stable across contexts). Compare to DreamBooth baseline.

#### 4.3.5 Diversity
- **LPIPS diversity**: Average pairwise LPIPS distance between generated images for the same subject-prompt pair. Higher = more diverse outputs.

#### 4.3.6 Efficiency
- **Training time** (seconds per subject on a single NVIDIA A100).
- **Model storage** (MB per subject).
- **Inference time** (seconds per image at 1024×1024 resolution).

#### 4.3.7 User Study
- Conduct a user study following DreamBooth's protocol: 100 participants, 50 comparative questions per participant.
- Evaluate: (a) subject fidelity, (b) prompt fidelity, (c) multi-subject coherence.
- Use majority voting to aggregate results.

### 4.4 Implementation Details

- **Backbone**: FLUX.1-dev (12B parameter DiT) as the primary architecture; Stable Diffusion 3 Medium as secondary for ablations.
- **LoRA configuration**: Rank adaptively assigned per block (r ∈ {4, 8, 16, 32}); LoRA alpha = rank; dropout = 0.05.
- **Training**: AdamW optimizer; learning rate 1×10⁻⁴ for LoRA parameters; batch size 1 with gradient accumulation of 4; 800 training steps per subject.
- **Prior preservation**: Generate 200 class-prior images using the frozen backbone before training; sample uniformly during training.
- **CCD loss**: Temperature τ = 0.07; K = 5 augmentations per reference image; segmentation via SAM-2.
- **Inference**: 30 DDPM steps; classifier-free guidance scale 7.5; negative attention strength γ = 3.0.
- **Hardware**: Single NVIDIA A100 (80 GB) for training; single NVIDIA RTX 4090 (24 GB) for inference.
- **Framework**: PyTorch 2.x with the `diffusers` library (HuggingFace).

### 4.5 Ablation Studies

We will ablate the following components to demonstrate the contribution of each:

#### Ablation A: Blockwise vs. Uniform LoRA
- **Condition 1**: Uniform LoRA rank across all DiT blocks (r = 8 everywhere).
- **Condition 2**: Our blockwise rank assignment.
- **Expected outcome**: Blockwise assignment achieves comparable or higher subject fidelity with fewer parameters and better prompt fidelity, because context blocks are not overwritten.

#### Ablation B: CCD Loss
- **Condition 1**: Training without CCD loss (standard DreamBooth + PPL only).
- **Condition 2**: Training with CCD loss.
- **Expected outcome**: CCD loss reduces the CAE metric by 20–30% and improves CLIP-T by 0.02–0.03, demonstrating reduced entanglement.

#### Ablation C: Informative Captioning
- **Condition 1**: Standard DreamBooth captions ("a [V] dog").
- **Condition 2**: LLM-generated informative captions.
- **Expected outcome**: Informative captions improve convergence speed (~30% fewer steps) and reduce entanglement, consistent with SID findings.

#### Ablation D: Masked Compositional Inference
- **Condition 1**: Naive LoRA addition for multi-subject generation (no masking).
- **Condition 2**: Token-aware masking without negative attention.
- **Condition 3**: Full protocol (masking + negative attention).
- **Expected outcome**: Progressive improvement in IIS and Composition Accuracy across conditions, with the full protocol achieving the best identity isolation.

#### Ablation E: Number of Reference Images
- Train with 1, 2, 3, 4, 5 reference images per subject.
- Measure DINO and CLIP-T for each.
- **Expected outcome**: Consistent with DreamBooth, 3–4 images should be optimal; our method should achieve higher fidelity at 1–2 images due to context augmentation.

#### Ablation F: LoRA Rank Sensitivity
- Vary identity-block rank ∈ {4, 8, 16, 32, 64}.
- **Expected outcome**: Diminishing returns beyond rank 16–32 for most subjects; rank 32 provides the best quality-storage trade-off.

### 4.6 Expected Results

#### 4.6.1 Single-Subject Personalization (DreamBooth Benchmark)

| Method | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ | CAE ↓ | Storage |
|--------|--------|----------|----------|-------|---------|
| DreamBooth (Imagen) | 0.696 | 0.812 | 0.306 | — | ~2 GB |
| DreamBooth (SD) | 0.668 | 0.803 | 0.305 | — | ~2 GB |
| Textual Inversion | 0.569 | 0.780 | 0.255 | — | ~5 KB |
| DreamBooth-LoRA | ~0.660 | ~0.800 | ~0.300 | Baseline | ~4 MB |
| **ModularBooth (Ours)** | **~0.710** | **~0.820** | **~0.315** | **−25%** | **~4 MB** |

We anticipate matching or exceeding DreamBooth-Imagen's subject fidelity on a DiT backbone, with notably improved prompt fidelity and reduced entanglement, at a fraction of the storage cost.

#### 4.6.2 Multi-Subject Composition

| Method | IIS ↑ | Comp. Acc. ↑ | DINO ↑ | CLIP-T ↑ |
|--------|-------|-------------|--------|----------|
| DreamBooth (naïve merge) | ~0.15 | ~35% | ~0.55 | ~0.27 |
| Custom Diffusion | ~0.25 | ~50% | ~0.58 | ~0.28 |
| TARA | ~0.35 | ~60% | ~0.62 | ~0.29 |
| LoRACLR | ~0.30 | ~55% | ~0.60 | ~0.29 |
| **ModularBooth (Ours)** | **~0.45** | **~72%** | **~0.65** | **~0.31** |

We expect ModularBooth to substantially outperform prior methods in identity isolation and composition accuracy, owing to the spatial masking protocol and contrastive disentanglement.

#### 4.6.3 Efficiency

| Method | Train Time | Storage/Subject | Inference Time |
|--------|-----------|----------------|----------------|
| DreamBooth (full FT) | ~5 min | ~2 GB | ~8 s |
| DreamBooth-LoRA | ~4 min | ~4 MB | ~8 s |
| ModularBooth (Ours) | ~5 min | ~4 MB | ~10 s |
| FreeCus (zero-shot) | 0 | 0 | ~8 s |

ModularBooth's training time is comparable to standard LoRA approaches. The slight inference overhead (~2 s) comes from the spatial masking computation, which is negligible in practice.

---

## 5. Discussion and Future Work

### 5.1 Potential Pitfalls

1. **Knowledge localization may not cleanly separate**: DiT blocks may not cleanly partition into "identity" and "context" blocks. We mitigate this by using soft rank assignment rather than binary block selection, and by validating the probing results across diverse subjects.

2. **CCD loss may overcorrect**: Aggressively disentangling context from appearance could cause the model to ignore legitimate context-dependent appearance changes (e.g., lighting effects on a shiny object). We address this by tuning $\lambda_2$ carefully and by applying CCD only to identity blocks.

3. **Segmentation quality**: The CCD loss and masked inference both depend on subject segmentation. SAM-2 is generally robust, but failure cases (transparent objects, subjects occluded by other objects) could propagate errors. We will report results stratified by segmentation quality.

4. **Scalability beyond 3 subjects**: Our masked inference protocol uses bounding boxes, which become increasingly constrained as the number of subjects grows. For >3 subjects, overlapping regions may cause degradation. We leave dense multi-subject composition to future work.

5. **Evaluation bias**: Automated metrics (DINO, CLIP) may not fully capture perceptual quality. We mitigate this through the user study and VQA-based evaluation.

### 5.2 Alternative Interpretations

- One might argue that zero-shot methods (FreeCus, Personalize Anything) render fine-tuning-based personalization obsolete. However, current zero-shot methods still struggle with fine-grained identity preservation for unusual or complex subjects. Our modular approach offers a complementary strategy: invest minutes of training per subject in exchange for substantially higher fidelity.
- The contrastive loss could alternatively be formulated as a mutual information minimization objective between subject and context features. We plan to explore this in ablations.

### 5.3 Future Directions

1. **Extension to video**: Apply ModularBooth's blockwise LoRA and CCD loss to video DiT architectures (e.g., CogVideoX, Wan) for temporally consistent subject-driven video generation.
2. **Extension to 3D**: Combine ModularBooth with score distillation sampling (SDS) to generate personalized 3D assets via Gaussian Splatting, leveraging the disentangled subject representations.
3. **Federated personalization**: Enable collaborative personalization where users contribute subject modules to a shared library without exposing raw images, addressing privacy concerns.
4. **Adaptive rank selection**: Replace the static probing-based rank assignment with a learnable rank allocation mechanism (e.g., using a hypernetwork that predicts optimal LoRA ranks per block given a subject's visual complexity).
5. **Preference alignment**: Integrate DreamBoothDPO's preference optimization with ModularBooth's disentanglement, allowing users to specify aesthetic preferences alongside identity constraints.
6. **Cross-architecture transfer**: Investigate whether LoRA modules trained on one DiT backbone (e.g., FLUX) can be transferred to another (e.g., SD3) via weight-space mapping, amortizing the per-subject training cost.

---

## 6. Conclusion

This research plan proposes **ModularBooth**, a principled extension of DreamBooth for modern Diffusion Transformer architectures that addresses three fundamental limitations of the original work: context-appearance entanglement, single-subject restriction, and full model fine-tuning overhead.

By combining blockwise-parameterized LoRA (localized to identity-critical DiT blocks), a contrastive context disentanglement loss (that explicitly decorrelates subject appearance from background), and a masked compositional inference protocol (that enables plug-and-play multi-subject generation), ModularBooth is expected to achieve state-of-the-art subject fidelity with improved prompt fidelity, reduced entanglement, and practical efficiency.

The proposed experimental design—covering single-subject benchmarks, a new multi-subject composition benchmark, six ablation studies, and a user study—provides rigorous validation of each contribution. The work is achievable with a single GPU, uses publicly available models and datasets, and addresses a practically important problem with clear scientific contributions suitable for a diploma thesis.

If successful, ModularBooth would demonstrate that the DreamBooth paradigm, far from being superseded by zero-shot approaches, can be modernized and modularized to deliver superior personalization quality with practical deployment characteristics.

---

## 7. References

1. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. *CVPR 2023*.

2. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A. H., Chechik, G., & Cohen-Or, D. (2022). An image is worth one word: Personalizing text-to-image generation using textual inversion. *arXiv:2208.01618*.

3. Kumari, N., Zhang, B., Zhang, R., Shechtman, E., & Zhu, J.-Y. (2023). Multi-concept customization of text-to-image diffusion. *CVPR 2023*.

4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685*.

5. Ye, H., Zhang, J., Liu, S., Han, X., & Yang, W. (2023). IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*.

6. DreamBoothDPO (2025). Improving personalized generation using direct preference optimization. *arXiv:2505.20975*.

7. DreamO (2025). A unified framework for image customization. *arXiv:2504.16915*.

8. Delta-SVD (2025). Efficient post-hoc compression for personalized models. *arXiv:2508.16863*.

9. MINDiff (2025). Mask-integrated negative attention for diffusion personalization. *arXiv:2511.17888*.

10. PPD — Personalized Preference Fine-tuning of Diffusion Models (2025). *arXiv:2501.06655*.

11. Beyond Fine-Tuning: A Systematic Study of Sampling Techniques in Personalized Image Generation (2025). *arXiv:2502.05895*.

12. DreamBlend (2025). Checkpoint blending for personalized diffusion models. *arXiv:2411.19390*.

13. Personalization Guidance (2025). *arXiv:2508.00319*.

14. Regularized Personalization without Distributional Drift (2025). *arXiv:2505.19519*.

15. FaceCLIP (2025). Learning joint ID-textual representation for ID-preserving image synthesis. *arXiv:2504.14202*.

16. TARA: Token-Aware LoRA for Composable Personalization in Diffusion Models (2025). *arXiv:2508.08812*.

17. DP-Adapter (2025). Dual-pathway adapter for identity-preserving generation. *arXiv:2502.13999*.

18. Preventing Shortcuts in Adapter Training (2025). *arXiv:2510.20887*.

19. SID — Selectively Informative Description (2024). *arXiv:2403.15330*.

20. Towards More Accurate Personalized Image Generation (2025). *arXiv:2503.06632*.

21. TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space (2025). *arXiv:2501.12224*.

22. LoRACLR: Contrastive Adaptation for Customization of Diffusion Models (2025). *CVPR 2025*.

23. Video Alchemist — Multi-subject Open-set Personalization in Video Generation (2025). *arXiv:2501.06187*.

24. LayerComposer: Multi-Human Personalized Generation via Layered Canvas (2025). *arXiv:2510.20820*.

25. BlockLoRA — Modular Customization via Blockwise-Parameterized Low-Rank Adaptation (2025). *arXiv:2503.08575*.

26. MuDI — Identity Decoupling for Multi-Subject Personalization (2024). *arXiv:2404.04243*.

27. Personalize Anything for Free with Diffusion Transformer (2025). *arXiv:2503.12590*.

28. FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers (2025). *ICCV 2025*.

29. Localizing Knowledge in Diffusion Transformers (2025). Zarei et al.

30. Conceptrol (2025). Improved IP-Adapter with textual concept masks. *arXiv:2503.06568*.

31. MONKEY (2025). Subject mask-aware IP-Adapter inference. *arXiv:2507.15249*.

32. SubZero (2025). Composition without fine-tuning. *arXiv:2502.19673*.

33. LoRA Diffusion: Zero-Shot LoRA Synthesis (2024). *arXiv:2412.02352*.

34. DynaIP: Dynamic Image Prompt Adapter (2025). *OpenReview*.

35. VideoBooth: Diffusion-based Video Generation with Image Prompts (2024). *ECCV 2024*.

36. CP-GS: Consistent 3D Scene Personalization via Gaussian Splatting (2025). *arXiv:2505.14537*.

37. Saharia, C., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding (Imagen). *arXiv:2205.11487*.

38. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models (Stable Diffusion). *CVPR 2022*.

39. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *arXiv:2103.00020*.

40. Caron, M., et al. (2021). Emerging properties in self-supervised vision transformers (DINO). *ICCV 2021*.

41. Kirillov, A., et al. (2023). Segment Anything. *ICCV 2023*.

42. Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR 2024*.
