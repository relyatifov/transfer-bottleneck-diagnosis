# Diagnosing the Transfer Bottleneck in a Model-Based Agent

Code and data for the paper **"Knowledge and Integration Deficits: Diagnosing the Transfer
Bottleneck in a Model-Based Agent"** (R. E. Lyatifov, Bauman Moscow State Technical University).

Russian title: *«Знаниевый и интеграционный дефицит: диагностика узкого места переноса
у агента с моделью мира»*, Neuroinformatics conference.

## What this is about

When a model-based agent fails on tasks that require a newly encountered element of the
environment (here: a lever that opens a bridge), the bottleneck is one of two things:

| Bottleneck | Meaning | Effective remedy | Cost |
|---|---|---|---|
| **Knowledge deficit** | the model is *wrong* about the element | collect new data | expensive |
| **Integration deficit** | the model is *right*, but the element never reaches behaviour | replan on the existing model | almost free |

Standard metrics (return, model accuracy) do not tell the two apart. This repository contains
a cheap diagnostic `K(X)` — a lower-confidence estimate of the learned effect of the element —
and an **interventional** validation showing that the remedy it prescribes is the one that
actually restores transfer.

## Headline result

Reproduced by `python experiment.py` (all numbers below are on a **held-out** split of agents;
the threshold θ is selected on a disjoint training split):

```
320 agents sampled; 292 fail at deployment
  replanning alone suffices in 49% of failures
  theta*(train) = 0.13
  TEST: AUC 0.913 [0.859; 0.960] | accuracy 0.863 [0.808; 0.918] | sens 0.96 spec 0.78
  prescribed fix recovers 88% of test failures
```

Ablations over the uncertainty coefficient `c`, the lever stochasticity `p_open`, and the
population size are printed by the same script.

## Reproducing

No GPU is required. The experiment is tabular and runs in a few minutes on a laptop CPU.

```bash
pip install -r requirements.txt
python experiment.py      # runs the full study + ablations, writes results.json and final.npz
python make_figures.py    # regenerates all four figures into figures/
```

All random seeds are fixed (`master_seed = 0`), so the numbers above are reproduced exactly.

## Files

| File | Contents |
|---|---|
| `experiment.py` | environment, tabular agent, diagnostic `K(X)`, train/test threshold selection, bootstrap CIs, ablations |
| `make_figures.py` | figures 1–4 of the paper |
| `results.json` | all reported numbers, including every ablation |
| `figures/` | generated figures |

## Method in brief

**Environment.** A 5×7 gridworld of two halls separated by a wall. A lever `X` opens a bridge
with probability `p_open = 0.7`. Goals live in the far hall and are unreachable without it.

**Population.** 320 agents with *independently drawn* budgets of data (exploration episodes)
and computation (value-iteration sweeps). The two deficit regimes are therefore **emergent**,
not hand-imposed.

**Diagnostic.**

```
K(X) = mean over (s,a) in D(X) of max(0, p_hat_sa - c / sqrt(n_sa))
```

where `D(X)` are the observed state-action pairs interacting with `X`, `p_hat` is the model's
predicted probability of the lever's effect, `n` the number of observations, and `c = 1` the
uncertainty coefficient. The penalty matters: without it, one lucky lever activation would
count as knowledge.

**Prescription rule.** `K(X) < θ` → collect data; otherwise → replan only.

**Validation.** Interventional, not correlational. Ground truth is defined by *which
intervention actually restores performance*, and the rule is scored on agents that took no
part in choosing θ.

## Key hyperparameters

| Parameter | Value |
|---|---|
| grid | 5 × 7, two halls |
| `p_open` | 0.7 |
| discount γ | 0.97 |
| horizon | 90 steps |
| exploration ε | 0.25 |
| data budgets | {2, 3, 4, 6, 10, 16, 24, 40, 70, 120} episodes |
| planning budgets | {1, 2, 3, 4, 6, 10, 25, 300} sweeps |
| data intervention | +90 episodes, then full recomputation |
| full recomputation | 300 sweeps |
| uncertainty coefficient `c` | 1.0 (ablated over 0.5–2.0) |
| failure / recovery thresholds | 0.5 / 0.7 |
| train/test split | 50 / 50 |
| bootstrap replicates | 2000 |

The complete dictionary lives in `HP` at the top of `experiment.py`.

## Limitations

Stated plainly, because they matter: the environment is tabular, small and fully observable;
`K(X)` is defined for a count-based model and extending it to neural world models requires an
ensemble or Bayesian uncertainty estimate; the integration deficit is induced by truncating the
planning budget, though the class is broader (stale value after a goal change, insufficient
search depth, an untransferred policy); mixed deficits, where both resources bind at once, are
not covered. Evaluation on richer benchmarks such as MiniGrid is ongoing work.

## Citation

```bibtex
@inproceedings{lyatifov2026bottleneck,
  title     = {Knowledge and Integration Deficits: Diagnosing the Transfer
               Bottleneck in a Model-Based Agent},
  author    = {Lyatifov, R. E.},
  booktitle = {Neuroinformatics},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
