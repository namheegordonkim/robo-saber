# Robo-Saber: Generating and Simulating VR Players

Official codebase for Robo-Saber (Eurographics 2026).

[Project page](https://robo-saber.github.io/)

This repository supports the scientific publication and released inference workflow. It is not intended to be a reusable Python library.

## Quick Start

Run this from the repository root:

```bash
uv run snakemake generate
```

This is the recommended one-command path. It creates or uses the uv project environment, runs `prepare` when required assets are missing, downloads the model bundle and data manifests, fetches missing BeatSaver maps and BoxRR replay tarballs as needed, and writes:

```text
out/gen3p.nc
```

The first run requires internet access. Generation currently assumes CUDA, so run inference on a machine with an NVIDIA GPU and a working CUDA-enabled PyTorch install.

## News

* **May 15, 2026:** Inference code released. Stay tuned for training code.
* May 7, 2026: Please star this codebase to be notified when the official code upload is done in the next few weeks.

## Status

* [x] Inference code
* [ ] Visualization code
* [ ] Training code

## Requirements

Recommended:

* [uv](https://docs.astral.sh/uv/)
* Internet access on the first run
* NVIDIA GPU with CUDA support for inference

The project is pinned to Python 3.11 through `pyproject.toml`.

## UV Workflow

The quick-start command is usually enough:

```bash
uv run snakemake generate
```

If you want to create the local `.venv` explicitly before running anything:

```bash
uv sync
```

To download only the publication assets:

```bash
uv run snakemake prepare
```

This downloads:

* `models/pretrained.pkl`
* `data/boxrr23_post_qc.csv`
* `data/heldout_player_maps.csv`
* `data/placeholder_3p.txt`
* `data/placeholder_3p_sixd.txt`

To override the default Snakemake inputs from `config.yaml`:

```bash
uv run snakemake generate --config csv_path='data/your_inputs.csv' target_player_source=csv
```

`config.yaml` currently points to `data/heldout_player_maps.csv` and sets `target_player_source: csv`.

## Conda Workflow

If you prefer Conda for environment setup:

```bash
conda env create -f environment.yml
conda run -n robo-saber --no-capture-output snakemake generate
```

The current Snakemake rules call `uv run` internally, so keep `uv` installed when using the Snakemake workflow. For a Conda-only Python execution path, use the manual script commands below.

## Manual Script Usage

Manual commands are useful when you want to run individual steps or change script arguments directly.

Prepare assets with uv:

```bash
uv run python robo-saber/prepare.py
```

Generate with uv:

```bash
uv run python robo-saber/generate.py \
  --csv_path data/heldout_player_maps.csv \
  --target_player_source csv
```

Generate with Conda:

```bash
conda run -n robo-saber --no-capture-output python robo-saber/generate.py \
  --csv_path data/heldout_player_maps.csv \
  --target_player_source csv
```

`generate.py` also accepts:

* `--boxrr23_manifest_path`, defaulting to `data/boxrr23_post_qc.csv`
* `--clean_models_bundle`, defaulting to `models/pretrained.pkl`
* `--nc_out_path`, defaulting to `out/gen3p.nc`

## Inputs and Outputs

The default publication inference run uses:

* `config.yaml` for Snakemake defaults
* `data/heldout_player_maps.csv` for requested player/map rows
* `data/boxrr23_post_qc.csv` as the BoxRR manifest
* `models/pretrained.pkl` as the single-file inference model bundle

During generation, missing BeatSaver maps are stored under `data/BeatSaver`, and player replay tarballs are fetched from the BoxRR Hugging Face dataset. Each successful output row is written as a NetCDF group in `out/gen3p.nc`.

## Validation

If `out/gen3p.gold.nc` is available, compare a generated file against it with:

```bash
uv run snakemake validate
```

or manually:

```bash
uv run python tools/compare_gen3p.py out/gen3p.gold.nc out/gen3p.nc
```

## Notes

* Run commands from the repository root.
* The released code is focused on inference for the publication.
* Training and visualization code are not yet included.
