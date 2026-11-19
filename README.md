# PoolAudit

Code and description pools for the paper *Separating Baseline Repair, Content Potential, and Pool Complementarity in CLIP Description-Pool Regeneration*.

PoolAudit regenerates sentence description pools for frozen vision-language classifiers and evaluates them in a controlled 2 x 2 design: flat attributes versus structured records, crossed with and without a text-only review round. Three evaluation views separate where a gain comes from: Direct accuracy (replacement), a label-using single-sentence diagnostic (content potential), and Union with the incumbent pool (complementarity).

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Pool generation, evaluation, ablations, seed repeats, second-generator checks, significance tests |
| `cupl_descriptions/` | Description pools for the four benchmarks: incumbent, regenerated arms, seed repeats, field ablations, unions |
| `results/` | Summary JSON files with the reported numbers |
| `vendor/clip/` | CLIP inference code and the BPE vocabulary (OpenAI CLIP, MIT) |
| `requirements.txt` | Python dependencies |

Datasets and image/feature caches are not included.

## Environment

```bash
python -m pip install -r requirements.txt
```

Place the benchmarks under `data/dataset/`. The root is configurable:

```bash
# Linux/macOS
export DATA_ROOT=/path/to/data/dataset
export CUB_ROOT=/path/to/data/dataset/cub/CUB_200_2011

# Windows PowerShell
$env:DATA_ROOT="E:\path\to\data\dataset"
$env:CUB_ROOT="E:\path\to\data\dataset\cub\CUB_200_2011"
```

## Main pipeline

Generation uses an OpenAI-compatible chat endpoint. The key is read from `--api-key`, `E5_API_KEY`, or `OPENAI_API_KEY`.

```bash
cd scripts

# 1. Generate one arm for one benchmark (onto / magent / omni)
python p4_b1_feas.py --dataset dtd --stage onto \
    --api-base https://api.openai.com/v1 --model <model-id>

# 2. Evaluate a generated arm against the incumbent
python p4_b1_feas.py --dataset dtd --stage eval

# 3. Paired significance tests on the same test images
python p5_stats.py --stage mcnemar

# 4. Union view: candidate pool concatenated with the incumbent (all four benchmarks)
python p5_union_mcnemar.py
```

The flat-attribute arm is generated with:

```bash
python e5_gen_pools.py --dataset dtd --arm attr --api-base https://api.openai.com/v1 --model <model-id>
```

Every other script exposes its options through `--help`, for example:

```bash
python p5_seeds.py --help
python p5_model2.py --help
python p5_cupl_arm.py --help
python p5_ablate_attrrel.py --help
```

## Evaluation views

Let `P_c` be the sentence pool of class `c`, `phi(d)` its frozen text feature, and `x` a normalized image feature. Class prototypes are the normalized means of their sentence features, and prediction is the nearest prototype.

- **Direct** — test accuracy of the mean-prototype rule, with no test-dependent prototype construction.
- **k1** — one sentence per class is selected with test labels while the other classes stay at their means; all selected columns are then replaced together. This is a diagnostic of content potential, not deployable accuracy and not a strict upper bound.
- **Union** — the same rule applied after concatenating the candidate and incumbent pools class by class, without selection or transduction.

## Notes

- `results/` holds the summary values reported in the paper; `cupl_descriptions/` holds the pools those numbers were computed from.
- `k1` uses test labels by construction. It must not be read as an accuracy that can be deployed.
- `vendor/clip/` is derived from OpenAI CLIP and is distributed under the MIT license; retain its copyright notice when redistributing.

