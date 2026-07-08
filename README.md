# The Multidimensional Geometry of Truth in LLMs

Modular research codebase for investigating whether propositional truth in LLMs is governed
by a single linear direction or by a higher-dimensional geometric structure (a *truth cone*).

The pipeline implements five experiments that, together, characterize the geometry of truth
representations in transformer residual streams: causal localization (Exp 1), gradient-based
direction discovery (Exp 2), multidimensional cone discovery (Exp 3), DIM/cone alignment
analysis (Exp 4), and retention of general capabilities (Exp 5).

## Architecture

```
src/
├── data/            # Factual + retain dataset loaders, prompt formatting
├── models/          # nnsight LanguageModel loading
├── interventions/   # Strategy pattern: ablation, addition, patching
│   ├── base.py      # Intervention interface
│   ├── ablation.py  # Surgical ablation across all layers
│   ├── addition.py  # α-scaled injection at a target layer
│   ├── patching.py  # Activation patching for Exp 1
│   ├── truth_cone.py# k-D orthonormal basis module
│   ├── tdo.py       # Truth Direction Optimization (1-D)
│   └── tco.py       # Truth Cone Optimization (k-D)
├── evaluation/      # ASR, KL retention, MC sampling, DIM alignment
├── persistence/     # Manifest writers, handoff JSON contract
└── utils/           # NPE, alpha calibration, visualization
```


## Usage

```bash
# Setup
pip install -e .

# Run the pipeline
python scripts/run_patching.py   --config configs/patching.yaml
python scripts/run_tdo.py        --config configs/tdo.yaml
python scripts/run_tco.py        --config configs/tco.yaml
python scripts/run_alignment.py  --config configs/alignment.yaml
python scripts/run_retention.py  --config configs/retention.yaml

```

## Output structure

Two-tier output system:

- `experimental_outputs/`: machine-readable artifacts (manifests, .pt weights, .npz matrices).
  Source of truth for downstream experiments.
- `figures/`: paper-ready visualizations generated from the manifests.

Each experiment writes a `results_summary.json` with a stable schema, which downstream
experiments and analysis consume. Heavy tensors live in `.pt` files alongside.

## References

- Marks & Tegmark (2024), *The Geometry of Truth*
- Wollschläger et al. (2025), *The Geometry of Refusal in LLMs*
- Arditi et al. (2024), *Refusal in Language Models is Mediated by a Single Direction*
