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

* [uv](https://docs.astral.sh/uv/): optional but recommended

## Installation

### With `uv` (Recommended)

```bash
uv run robo-saber/prepare.py
```

This downloads the pretrained model snapshot and post-QC manifest for BOXRR-23.

### Without `uv`

#### 1. Create a `venv`, `conda` environment, or similar virtual environment to your liking. The codebase is tested on Python 3.11.

Install the requirements

```bash
pip install -r requirements.txt
```

#### 2. Download the pretrained model snapshot

#### 3. Download the post-QC manifest for BOXRR-23

### `conda`

### `venv`

### `pip`

## Running the code

### With `uv` (Recommended)

```bash
uv run robo-saber/generate.py
```

### Without `uv`

```bash
python 
```