# Robo-Saber: Generating and Simulating VR Players

Official codebase for Robo-Saber (Eurographics 2026).

[Project page](https://robo-saber.github.io/)

## Quick Start

Run this from the repository root:

```bash
pixi run snakemake generate
```

This is the recommended one-command path. It creates or uses the Pixi project environment, runs `prepare` when required assets are missing, downloads the model bundle and data manifests, fetches missing BeatSaver maps and BOXRR-23 replay tarballs as needed, and writes:

```text
out/gen3p.nc
```

The first run requires internet access. Generation currently assumes CUDA, so run inference on a machine with an NVIDIA GPU and a working CUDA-enabled PyTorch install.

## News

* **September 25, 2026:** Thanks for waiting! I'm now working on releasing the training and visualization code soon.
* **May 15, 2026:** Inference code released. Stay tuned for training code.
* May 7, 2026: Please star this codebase to be notified when the official code upload is done in the next few weeks.

## Status

* [x] Inference code
* [ ] Visualization code
* [x] Training code

## Requirements

Recommended:

* [Pixi](https://pixi.sh/latest/installation/)
* Linux x86-64 (the configured Pixi platform)
* Internet access on the first run
* NVIDIA GPU with CUDA support for inference

The Pixi environment is pinned to Python 3.11.13 through `pyproject.toml`. Python dependencies remain in `requirements.txt`, and `pixi.lock` records the resolved environment.

## Pixi Workflow

The quick-start command is usually enough:

```bash
pixi run snakemake generate
```

If you want to create the local `.pixi/envs/default` environment explicitly before running anything:

```bash
pixi install --locked
```

To download only the publication assets:

```bash
pixi run snakemake prepare
```

This downloads:

* `models/pretrained.pkl`
* `data/boxrr23_post_qc.csv`
* `data/heldout_player_maps.csv`
* `data/placeholder_3p.txt`
* `data/placeholder_3p_sixd.txt`

To override the default Snakemake inputs from `config.yaml`:

```bash
pixi run snakemake generate --config csv_path='data/your_inputs.csv' target_player_source=csv
```

`config.yaml` currently points to `data/heldout_player_maps.csv` and sets `target_player_source: csv`.

## Conda Workflow

If you prefer Conda for environment setup:

```bash
conda env create -f environment.yml
conda run -n robo-saber --no-capture-output snakemake generate
```

The Snakemake rules use Python from the active environment.

## Manual Script Usage

Manual commands are useful when you want to run individual steps or change script arguments directly.

Prepare assets with Pixi:

```bash
pixi run python robo-saber/prepare.py
```

Generate with Pixi:

```bash
pixi run python robo-saber/generate.py \
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
* `data/boxrr23_post_qc.csv` as the BOXRR-23 manifest
* `models/pretrained.pkl` as the single-file inference model bundle

During generation, missing BeatSaver maps are stored under `data/BeatSaver`, and player replay tarballs are fetched from the BOXRR-23 Hugging Face dataset. Each successful output row is written as a NetCDF group in `out/gen3p.nc`.

## Training

From the repository root:

```bash
pixi run snakemake train
```

The `train` rule runs `robo-saber/train.py` after `prepare` supplies the pretrained bundle, BOXRR-23 manifest, held-out player/map split, and 6D placeholder. It tracks `out/train/checkpoint.pkl` and `out/train/pretrained.pkl` as outputs. Training requires internet access throughout the run to stream replays and fetch maps; CUDA is used when available, otherwise CPU.

To inspect the selected rules, dependencies, and command without starting training:

```bash
pixi run snakemake train --dry-run --printshellcmds
```

Add `--keep-incomplete` to the training command to retain checkpoints if the Snakemake job fails or is interrupted. Rerunning `snakemake train` starts fresh with the default profile; use the manual `--checkpoint` command below to resume.

The script can also be run directly with `pixi run python robo-saber/train.py`.

This trains the existing conditional predictor and motion VAE for 500,000 updates with an effective batch size of 128, float32, AdamW at a constant `5e-5`, reconstruction MSE plus `1e-4 × JSD`, and EMA. There are no training rollouts. The released player classifier is reused unchanged for inference exports.

Publication assets are prepared automatically when missing. The loader streams player archives from [BOXRR-23](https://huggingface.co/datasets/cschell/boxrr-23/tree/af25be3ef76b176fbee0a094e82d97a611f9c950) using Hugging Face iterable worker sharding. It matches replay metadata to post-QC, excludes all listed holdout players and map difficulties from training, and validates on the holdout/post-QC intersection. No extra popularity filter is applied.

Workers automatically download matching map versions through [BeatSaver's hash API](https://api.beatsaver.com/docs/swagger.json), reuse `data/BeatSaver`, and keep new downloads in private temporary caches of at most 512 MiB each. Oversized maps are used once and discarded. Missing or corrupt records are reported and skipped. Each player block contains at most 64 accepted replays, with 32 sampled windows per replay; six same-player windows form an example. Completed examples use a 1,024-example shuffle buffer. Four workers prefetch two batches each by default.

`out/train/checkpoint.pkl` includes live/EMA weights, optimizer, progress, RNG, and fixed evaluation batches. It is saved initially, every 25,000 updates, at completion, and on Ctrl-C. `out/train/pretrained.pkl` is the compatible inference bundle, containing live generator weights and the released classifier. Fixed train/validation evaluations run initially, every 1,250 updates, and at completion. Metrics and loader timing are written to the console and local TensorBoard logs.

```bash
pixi run python robo-saber/train.py --checkpoint out/train/checkpoint.pkl
pixi run tensorboard --logdir out/train/tensorboard
pixi run python robo-saber/generate.py --clean_models_bundle out/train/pretrained.pkl
```

Operational options are `--checkpoint`, `--out-dir`, `--total-batches` (the total target, including resumed updates), `--microbatch-size` (default 16), and `--workers` (default 4, capped by available CPUs). Reduce the microbatch size to fit GPU memory; the optimizer batch remains 128. Resume restores model/optimizer/RNG state and restarts streaming with a new deterministic pass seed. Prefetched records are not restored. This reproduces the released architecture and optimization recipe; the reconstructed split and streaming sampler do not reproduce the historical run bit for bit.

## Notes

* Run commands from the repository root.
* Visualization code is not yet included.
