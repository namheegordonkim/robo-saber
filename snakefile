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
DATA_DIR = config.get("data_dir", "/home/nhgk/projects/Beaty/out/1654/002_main_data/")
REFERENCE_SOURCE = config.get("reference_source", "huggingface")
TARGET_PLAYER_SOURCE = config.get("target_player_source", "strong_random")


rule all:
    default_target: True
    input:
        "out/gen3p.nc"


rule prepare:
    output:
        ccm=PREPARE_PATHS["ccm.pkl"],
        classy=PREPARE_PATHS["classy.pkl"],
        phc=PREPARE_PATHS["phc.pkl"],
        boxrr23_manifest=PREPARE_PATHS["boxrr23_post_qc.csv"]
    message:
        "Preparing Robo-Saber model files and BOXRR-23 manifest."
    shell:
        "uv run robo-saber/prepare.py"


rule generate:
    input:
        ccm=rules.prepare.output.ccm,
        classy=rules.prepare.output.classy,
        boxrr23_manifest=rules.prepare.output.boxrr23_manifest
    output:
        "out/gen3p.nc"
    params:
        csv_path=CSV_PATH,
        data_dir=DATA_DIR,
        reference_source=REFERENCE_SOURCE,
        target_player_source=TARGET_PLAYER_SOURCE
    message:
        "Generating 3P trajectories and writing out/gen3p.nc."
    shell:
        "uv run robo-saber/generate.py "
        "--csv_path {params.csv_path:q} "
        "--data_dir {params.data_dir:q} "
        "--reference_source {params.reference_source:q} "
        "--target_player_source {params.target_player_source:q} "
        "--gen_path {input.ccm:q} "
        "--classy_path {input.classy:q} "
        "--boxrr23_manifest_path {input.boxrr23_manifest:q}"
