# Robo-Saber: Generating and Simulating VR Players
Official codebase for Robo-Saber (Eurographics 2026)

[Project page](https://robo-saber.github.io/)

## News
* **May 15, 2026:** Inference code released. Stay tuned for training code.
* May 7, 2026: Please star this codebase to be notified when the official code upload is done in the next few weeks.

## TODO
* [x] Inference code
* [ ] Visualization code
* [ ] Training code

## Dependencies

* [uv](https://docs.astral.sh/uv/)
* Internet access on the first run so `prepare.py` can fetch the model bundle and manifests, and `generate.py` can fetch missing BeatSaver maps and replay tarballs

## Installation

```bash
uv sync
```

This creates a local `.venv` with Python 3.11 and installs the repo from `pyproject.toml`.

## Prepare assets

```bash
uv run snakemake prepare
```

This downloads:

* `models/pretrained.pkl`
* `data/boxrr23_post_qc.csv`
* `data/heldout_player_maps.csv`
* `data/placeholder_3p.txt`
* `data/placeholder_3p_sixd.txt`

## Running the code

Recommended:

```bash
uv run snakemake generate
```

This writes `out/gen3p.nc`.

To override the default Snakemake inputs from `config.yaml`:

```bash
uv run snakemake generate --config csv_path='data/your_inputs.csv' target_player_source=csv
```

Direct script usage:

```bash
uv run robo-saber/generate.py \
  --csv_path data/heldout_player_maps.csv \
  --target_player_source csv
```

`snakemake generate` uses the same defaults from `config.yaml`.

## Notes

* Run commands from the repository root.
* `generate.py` currently assumes CUDA. Run inference on a machine with an NVIDIA GPU and a working CUDA-enabled PyTorch install.
* `config.yaml` controls the default `snakemake generate` inputs.
