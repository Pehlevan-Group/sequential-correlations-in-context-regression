# Sequential Correlations Change In-Context Learning

This repository contains all code needed to reproduce figures and experiments for our upcoming paper "Sequential Correlations Change In-Context Learning: Effective Context Length and Architectural Mismatch" from _Mary Letey, Yue M. Lu, Cengiz Pehlevan, and Jacob Zavatone-Veth._ Paper to be released shortly.

## Repo organisation
This repository will be organised as follows

- `theory_base`: all code for running simulations of the theory model, i.e. computing the reduced-linear-attention parameter matrix $\Gamma^*$ from data.
- `transformer_base`: all basic architecture specs and training code for the models we train, i.e. full parameter linear attention and various softmax / mlp architectures.
- `specific_figures`: saved data from our runs that generate our figures, as well as instructions for regenerating this data from scratch using `theory_base` and `transformer_base`.

## Environment

This repository uses [`uv`](https://docs.astral.sh/uv/) to create a Python environment. Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment with NVIDIA GPU support. The command below names the environment directory `corr-lin-icl-env` (rename however you'd like).

```bash
uv python install 3.11
UV_PROJECT_ENVIRONMENT=corr-lin-icl-env uv sync --extra cuda12
source corr-lin-icl-env/bin/activate
```

You can use `--extra cuda13` instead of `--extra cuda12` if your machine has a new enough NVIDIA driver for CUDA 13. For CPU-only local development, omit the CUDA extra:

```bash
UV_PROJECT_ENVIRONMENT=corr-lin-icl-env uv sync
source corr-lin-icl-env/bin/activate
```

After activating the environment, check that JAX sees the expected device:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

If you plan to run the notebooks, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name corr-lin-icl-env --display-name "Python (corr-lin-icl-env)"
```

Some figure scripts currently import helper modules from sibling directories. Before running those scripts from the repository root, set:

```bash
export PYTHONPATH="$PWD/theory_base:$PWD/specific_figures/FIGURE1_compare_query_and_effective_samples:${PYTHONPATH:-}"
```
