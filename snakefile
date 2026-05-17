from pathlib import Path
import tomllib

configfile: "config.yaml"

with open("pyproject.toml", "rb") as f:
    PYPROJECT = tomllib.load(f)
PREPARE_PATHS = {
    Path(item["path"]).name: item["path"]
    for item in PYPROJECT["tool"]["prepare"]["downloads"]
}

CSV_PATH = config.get("csv_path", "data/for_video.csv")
TARGET_PLAYER_SOURCE = config.get("target_player_source", "strong_random")


rule all:
    default_target: True
    input:
        "out/gen3p.nc"


rule prepare:
    output:
        pretrained=PREPARE_PATHS["pretrained.pkl"],
        boxrr23_manifest=PREPARE_PATHS["boxrr23_post_qc.csv"],
        heldout_maps=PREPARE_PATHS["heldout_player_maps.csv"],
        placeholder_sixd=PREPARE_PATHS["placeholder_3p_sixd.txt"],
        placeholder_3p=PREPARE_PATHS["placeholder_3p.txt"],
    message:
        "Downloading pretrained model and data manifests from prepare.py (Google Drive)."
    shell:
        "uv run robo-saber/prepare.py"


rule generate:
    input:
        pretrained=rules.prepare.output.pretrained,
        boxrr23_manifest=rules.prepare.output.boxrr23_manifest
    output:
        "out/gen3p.nc"
    params:
        csv_path=CSV_PATH,
        target_player_source=TARGET_PLAYER_SOURCE,
    message:
        "Generating 3P trajectories and writing out/gen3p.nc."
    shell:
        "uv run robo-saber/generate.py "
        "--csv_path {params.csv_path:q} "
        "--target_player_source {params.target_player_source:q} "
        "--clean_models_bundle {input.pretrained:q} "
        "--boxrr23_manifest_path {input.boxrr23_manifest:q}"
