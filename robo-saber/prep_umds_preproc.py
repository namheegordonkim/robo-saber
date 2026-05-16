import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from proj_utils import my_logging
from proj_utils.dirs import proj_dir

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12346,
            stdout_to_server=True,
            stderr_to_server=True,
            suspend=False,
        )

    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")

    outdir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    umds_df = pd.read_csv(f"{proj_dir}/data/boxrr_umds_v17.csv")  # Must assume this exists for now
    umds_df["Normalized Score"] = umds_df.groupby(["User ID", "Song Hash and Difficulty"])["Normalized Score"].transform("mean")
    heldout_maps = pd.read_csv(f"{proj_dir}/{args.split_dir}/heldout_maps.csv")
    heldout_players = pd.read_csv(f"{proj_dir}/{args.split_dir}/heldout_players.csv")
    N = pd.read_csv(f"{proj_dir}/{args.split_dir}/N.csv")
    R = pd.read_csv(f"{proj_dir}/{args.split_dir}/R.csv")

    main_yes = ~umds_df["User ID"].isin(heldout_players["User ID"]) & ~umds_df["Song Hash and Difficulty"].isin(heldout_maps["Song Hash and Difficulty"])
    heldout_yes = umds_df["User ID"].isin(heldout_players["User ID"]) & umds_df["Song Hash and Difficulty"].isin(heldout_maps["Song Hash and Difficulty"])
    etc_yes = umds_df["User ID"].isin(heldout_players["User ID"]) & ~umds_df["Song Hash and Difficulty"].isin(heldout_maps["Song Hash and Difficulty"])
    # umds_df = umds_df[main_yes | heldout_yes].reset_index(drop=True)
    umds_df.loc[main_yes, "type"] = "main"
    umds_df.loc[heldout_yes, "type"] = "heldout"
    umds_df.loc[etc_yes, "type"] = "etc"

    main_df = umds_df[umds_df["type"] == "main"]
    heldout_df = umds_df[umds_df["type"] == "heldout"]

    assert ~(main_df["User ID"].isin(heldout_players["User ID"]).any())
    assert ~(main_df["Song Hash and Difficulty"].isin(heldout_maps["Song Hash and Difficulty"]).any())

    logger.info(f"{umds_df.shape=}")
    logger.info(f"{main_df.shape=}")
    logger.info(f"{heldout_df.shape=}")

    umds_df = umds_df[main_yes | heldout_yes | etc_yes].reset_index(drop=True)
    if args.main_no:
        umds_df = umds_df[umds_df["type"] != "main"].reset_index(drop=True)

    np.random.seed(0)

    # Shuffle the rows
    shuffle_idxs = np.random.permutation(umds_df.shape[0])
    umds_df = umds_df.iloc[shuffle_idxs].reset_index(drop=True)

    # Populate the indices for sampling
    idxs = np.random.choice(len(umds_df), 3_000_000)

    # Optionally subsample
    subsample_idx = np.arange(max(int(idxs.shape[0] * args.task_scale), 1))
    idxs = idxs[subsample_idx]

    # For job arrays, split according to job array indices
    this_job_idxs = np.array_split(idxs, args.job_array_size)[args.job_array_i]

    # Mandatory elements: at least one main and at least one heldout from N and R each
    npr_st = np.random.get_state()
    np.random.seed(args.job_array_i)  # Ensure different jobs get different samples
    main_idxs = np.where(umds_df["type"] == "main")[0]
    include_me = np.random.choice(main_idxs, 1)
    this_job_idxs = np.concatenate([this_job_idxs, include_me])

    heldout_idxs = np.where(umds_df["type"] == "heldout")[0]
    include_me = np.random.choice(heldout_idxs, 1)
    this_job_idxs = np.concatenate([this_job_idxs, include_me])

    N_idxs = pd.merge(umds_df.reset_index(), N, on=["User ID", "Song Hash and Difficulty"], how="inner")["index"].values
    include_me = np.random.choice(N_idxs, 1)
    this_job_idxs = np.concatenate([this_job_idxs, include_me])

    R_idxs = pd.merge(umds_df.reset_index(), R, on=["User ID", "Song Hash and Difficulty"], how="inner")["index"].values
    include_me = np.random.choice(R_idxs, 1)
    this_job_idxs = np.concatenate([this_job_idxs, include_me])
    np.random.set_state(npr_st)

    # Group by User ID
    grouped = umds_df.groupby("User ID")
    dfs = {id: group for id, group in grouped}
    group_idxs = []
    ids = []
    types = []
    # idxs = np.sort(np.random.choice(len(umds_df), 1_000_000))
    pbar = tqdm(total=this_job_idxs.shape[0] * args.n_segs)
    for i in this_job_idxs:
        id = umds_df.iloc[i]["User ID"]
        df = dfs[id]
        cand_idxs = np.arange(len(df))
        sampled_idx = np.random.choice(cand_idxs, args.n_segs, replace=True)
        df_ = df.iloc[sampled_idx].copy().reset_index(drop=True)
        df_ = df_[["Shard Index", "Datapoint Index"]]
        group_idxs.append(df_.values)
        ids.append(id)
        types.append(umds_df.iloc[i]["type"])
        pbar.update(df_.shape[0])
    pbar.close()

    stacked = np.stack(group_idxs, 0)
    ids = np.array(ids)
    types = np.array(types)

    # shuffle_idxs = np.random.permutation(len(stacked))
    # stacked = stacked[shuffle_idxs]

    out_path = f"{outdir}/part_{args.job_array_i:03d}.npz"
    np.savez_compressed(out_path, stacked, ids, types, fmt="%d")

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--debug_yes", "-d", action="store_true")  # if set, will pause the program
    parser.add_argument("--job_array_size", type=int, default=1)
    parser.add_argument("--job_array_i", type=int, default=0)
    parser.add_argument("--n_segs", type=int, required=True)
    parser.add_argument("--task_scale", type=float, default=1.0)
    parser.add_argument("--main_no", action="store_true")
    # parser.add_argument("--type", type=str, choices=["main", "heldout"], required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
