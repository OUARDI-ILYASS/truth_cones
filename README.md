# The Multidimensional Geometry of Truth in Large Language Models

**A Concept-Cone Extension of the Linear Representation Hypothesis**

Ilyass Ouardi — Università degli Studi di Milano, Department of Computer Science

---

## The question

The Linear Representation Hypothesis says high-level concepts are encoded as linear directions in a model's residual stream. For propositional truth — the model deciding whether "The city of Tokyo is in Japan" is True or False — the standard picture is a single truth direction.

**Is one direction enough? Or does truth occupy a higher-dimensional *cone* of directions — and if so, at what dimension *k*, and at what cost to the model's other abilities?**

## The answer

For the six instruction-tuned models tested (Qwen-2.5 1.5/7/14B, Gemma-2 2/9B, Llama-3.1-8B):

- **Below ~7B parameters, a single direction suffices.** Both causally effective and surgical.
- **At or above ~7B, a 2-D cone is needed.** A 1-D direction fails on two axes at once — it can't reliably flip the verdict (low ablation ASR) and it damages unrelated prompts (KL > 0.1). A 2-D cone closes both gaps simultaneously.
- **The cone is not DIM with noise.** At k=4 every basis vector is near-orthogonal to the DIM probe (|cos| < 0.1) yet the cone still flips the model — genuinely new structure.
- **Higher k is not free.** At k=3, surgicality can regress. k=2 is the sweet spot.

This extends the Linear Representation Hypothesis rather than breaking it: truth lives in a linear subspace, just one of dimension two, not one.

## Method

Four stages, each consuming the previous stage's saved artifact:

```
Activation Patching  →  DIM baseline  →  TDO (1-D)  →  TCO (k-D cone)  →  Evaluation
     §3.2                  §3.3            §3.4           §3.5              §3.6
  picks l*            θ_DIM, α          direction r     basis V         ASR, KL
```

**Activation patching (§3.2).** Denoising sweep over (layer, position) with Symmetric Token Replacement (STR). Selects l* as the most-downstream layer in NPE group (b) with mean NPE > 0.1. 50 contrastive pairs × 3 datasets (cities, animals, elements).

**DIM baseline (§3.3).** θ_DIM = μ₊ − μ₋ at (l*, −5). The direction between the true and false class means. Its norm α = ‖θ_DIM‖₂ calibrates the addition magnitude to the model's own representational scale.

**Truth Direction Optimization — TDO (§3.4, Algorithm 1).** Refines the DIM warm-start under a composite loss encoding both causal axioms plus a KL-retention term:

    L(r) = λ_abl · CE(f_ablate(r)(p_true), t_false)           — necessity
         + λ_add · CE(f_add(r,l*,α)(p_false), t_true)         — sufficiency
         + λ_ret · KL(f_ablate(r)(p_retain) ‖ f(p_retain))    — surgicality

with (λ_abl, λ_add, λ_ret) = (1.0, 0.2, 1.0). Gradient projected onto the tangent plane of S^(d−1) after each step; r is renormalized to stay on the unit sphere.

**Truth Cone Optimization — TCO (§3.5, Algorithm 2).** Extends TDO to a k-dimensional orthonormal basis V = [v₁, …, vₖ] with two additions: modified Gram–Schmidt after every gradient step (V^T V = Iₖ), and Monte-Carlo sampling of interior directions so the loss constrains the entire cone, not just its boundary. Warm-start is incremental: the k-cone reuses the (k−1)-cone plus one fresh random vector.

**Evaluation (§3.6).** MC ASR (32 directions sampled from the cone interior) for necessity and sufficiency; mean MC KL on 100 filtered Alpaca instructions (30 token positions) for surgicality. Threshold: KL < 0.1 (Arditi et al. 2024).

## Repository structure

```
├── configs/                   YAML configs for each pipeline stage
│   ├── patching.yaml
│   ├── tdo.yaml
│   ├── tco.yaml
│   ├── alignment.yaml
│   └── retention.yaml
│
├── scripts/                   CLI drivers (one per stage)
│   ├── run_patching.py        §3.2  activation patching → layer_selection.json
│   ├── run_tdo.py             §3.4  TDO → weights.pt (1-D direction)
│   ├── run_tco.py             §3.5  TCO → weights.pt (k-D cone basis)
│   ├── run_alignment.py       §4.3  DIM–cone cosine analysis
│   └── run_retention.py       §4.4  KL retention evaluation
│
├── src/
│   ├── config.py              Pydantic schemas for all hyperparameters
│   ├── logging_setup.py       Centralized logging
│   │
│   ├── interventions/
│   │   ├── projection.py      proj_r̂(x) = (r̂ᵀx)r̂  — the one primitive
│   │   ├── ablation.py        Eq. 1: x̃ = x − r̂r̂ᵀx  ∀ l, i
│   │   ├── addition.py        Eq. 2: x̃ = x + α·r    at l*
│   │   ├── truth_cone.py      TruthCone module (basis, Gram–Schmidt, sampling)
│   │   ├── tdo.py             Algorithm 1 training loop
│   │   ├── tco.py             Algorithm 2 training loop (optimized variant)
│   │   ├── patching.py        Denoising activation patching sweep
│   │   └── base.py            Abstract Intervention (Strategy pattern)
│   │
│   ├── evaluation/
│   │   ├── asr.py             Answer Switching Rate (necessity + sufficiency)
│   │   ├── kl.py              KL retention (surgicality) + kl_div_fn
│   │   ├── mc_sampling.py     Monte-Carlo ASR over cone interior
│   │   └── alignment.py       |cos(θ_DIM, vᵢ)| analysis
│   │
│   ├── models/
│   │   └── loader.py          nnsight LanguageModel loading + Gemma workarounds
│   │
│   ├── data/
│   │   ├── prompts.py         Few-shot prompt formatting
│   │   ├── factual.py         Factual dataset loading
│   │   ├── retain.py          Alpaca retain-prompt filtering
│   │   └── tdo_dataset.py     (p_true, p_false, p_retain) triples + collate
│   │
│   ├── persistence/
│   │   ├── handoff.py         layer_selection.json read/write (l* handoff)
│   │   ├── manifest.py        Timestamped JSON + .pt file management
│   │   └── results_summary.py Summary builders for each pipeline stage
│   │
│   └── utils/
│       ├── npe.py             NPE metric + l* selection logic
│       ├── alpha.py           DIM extraction + α = ‖θ_DIM‖₂ calibration
│       └── visualization.py   NPE heatmaps + training curves
│
├── datasets/
│   ├── cities.csv / cities_contrastive.csv
│   ├── animals.csv / animals_contrastive.csv
│   ├── elements.csv / elements_contrastive.csv
│   └── alpaca_data.json       Retain prompts (filtered of factual keywords)
│
├── experimental_outputs/      Saved artifacts (one dir per stage)
│   ├── exp1_patching/         NPE .npz matrices + layer_selection.json
│   ├── exp2_tdo/              TDO weights.pt + results_summary.json
│   ├── exp3_tco/              TCO weights.pt per (model, k) + results_summary.json
│   ├── exp4_alignment/        results_summary.json (cosine similarities)
│   └── exp5_retention/        results_summary.json (MC KL per cone)
│
├── paper/
│   ├── paper_revised.tex      LaTeX source
│   ├── tables/                All paper tables (tab_master, tab_cone_asr, …)
│   └── figures/               Teaser, pipeline, NPE heatmap figures
│
└── config.ini                 Model weight paths (maps names to local cache)
```

## Running the pipeline

**Prerequisites.** Python 3.11+, PyTorch, nnsight, transformers, bitsandbytes. A single GPU with ≥40GB VRAM (H100 80GB used for the paper). Model weights downloaded via HuggingFace (set `RUNPOD_HF_TOKEN` in environment).

```bash
# 1. Localize the truth-mediating layer (§3.2)
python scripts/run_patching.py --config configs/patching.yaml

# 2. Train a 1-D truth direction (§3.4)
python scripts/run_tdo.py --config configs/tdo.yaml

# 3. Train k-D truth cones (§3.5)
python scripts/run_tco.py --config configs/tco.yaml

# 4. DIM–cone alignment analysis (§4.3)
python scripts/run_alignment.py --config configs/alignment.yaml

# 5. KL retention evaluation (§4.4)
python scripts/run_retention.py --config configs/retention.yaml
```

Each stage reads its predecessor's artifacts and writes its own to `experimental_outputs/`. Run one model at a time with `--model <name>`.

## Key hyperparameters

All held fixed across models (from Wollschläger et al. 2025, Tab. 3):

| Parameter | Value | Source |
|---|---|---|
| Loss weights (λ_abl, λ_add, λ_ret) | (1.0, 0.2, 1.0) | Tab. hyperparams |
| Learning rate | 10⁻² (AdamW) | §8.4 |
| Physical / effective batch | 1 / 16 | Gradient accumulation |
| MC directions (training) | 16 per step | §3.5 |
| MC directions (evaluation) | 32 per cone | §3.6 |
| Retain prompts | 100 filtered Alpaca | §3.6 |
| KL window | last 30 token positions | §3.6 |
| Surgical threshold | KL < 0.1 | Arditi et al. 2024 |
| Cone dimensions tested | k ∈ {1, 2, 3, 4} | §4.2 |

## Models tested

| Model | Family | Params | l* depth | k* |
|---|---|---|---|---|
| Qwen-2.5-1.5B-Instruct | Qwen | 1.5B | mid | 1 |
| Gemma-2-2B-IT | Gemma | 2B | mid | 1 |
| Qwen-2.5-7B-Instruct | Qwen | 7B | mid | 2 |
| Llama-3.1-8B-Instruct | Llama | 8B | 0.44 | 2 |
| Gemma-2-9B-IT | Gemma | 9B | mid | 2 |
| Qwen-2.5-14B-Instruct | Qwen | 14B | mid | 2 |

All loaded in bfloat16 via nnsight on a single H100 80GB (§4.1). Gemma-2 models use `attn_implementation="eager"` to avoid NaN from logit soft-capping under the default sdpa path.

## References

- Azaria & Mitchell (2023). *The Internal State of an LLM Knows When It's Lying.*
- Marks & Tegmark (2024). *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.*
- Wollschläger et al. (2025). *The Geometry of Refusal in Large Language Models.*
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
- Park et al. (2024). *The Linear Representation Hypothesis and the Geometry of Large Language Models.*
- Zhang et al. (2024). *Best Practices for Activation Patching.*
- Meng et al. (2023). *Locating and Editing Factual Associations in GPT.*
- Heimersheim & Janiak (2024). *How to Use and Interpret Activation Patching.*

## License

This repository accompanies an academic project. Code is provided for reproducibility. Contact the author for licensing questions.