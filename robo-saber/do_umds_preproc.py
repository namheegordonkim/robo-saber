import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
import torch
import glob

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from beaty_common.bsmg_xror_utils import open_beatmap_from_unpacked_xror, load_cbo_and_3p
from beaty_common.data_utils import SegmentSampler
from dash import dash
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from beaty_common.train_utils import BoxrrHFDataset, nanpad_collate_fn
from xror.xror import XROR

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

    np.random.seed(0)

    umds_df = pd.read_csv(f"{proj_dir}/data/boxrr_umds_v17.csv")
    umds_df["Normalized Score"] = umds_df.groupby(["User ID", "Song Hash and Difficulty"])["Normalized Score"].transform("mean")

    npz_loaded = np.load(glob.glob(f"{proj_dir}/{args.manifest_dir}/**/part_000.npz", recursive=True)[-1])
    index_pair_sets = npz_loaded["arr_0"]
    ids = npz_loaded["arr_1"]
    types = npz_loaded["arr_2"]

    global_total = (index_pair_sets.shape[0] + (args.job_array_size * 2)) * index_pair_sets.shape[1]
    this_job_idxs = np.array_split(np.arange(index_pair_sets.shape[0]), args.job_array_size)[args.job_array_i]

    # Mandatory elements: at least one main and at least one heldout from N and R each
    npr_st = np.random.get_state()
    np.random.seed(args.job_array_i)  # Ensure different jobs get different samples
    main_idxs = np.where(types == "main")[0]
    include_me = np.random.choice(main_idxs, 1)
    this_job_idxs = np.concatenate([this_job_idxs, include_me])

    heldout_idxs = np.where(types == "heldout")[0]
    include_me = np.random.choice(heldout_idxs, 1)
    this_job_idxs = np.concatenate([this_job_idxs, include_me])
    np.random.set_state(npr_st)

    index_pair_sets = index_pair_sets[this_job_idxs]
    ids = ids[this_job_idxs]
    types = types[this_job_idxs]

    # shuffle_idxs = np.random.permutation(ids.shape[0])
    # index_pair_sets = index_pair_sets[shuffle_idxs]
    # ids = ids[shuffle_idxs]
    # types = types[shuffle_idxs]

    segment_sampler = SegmentSampler()
    minibatch_size = 1
    segment_length = 72
    stride = 4
    purview_notes = 20
    floor_time = -0.1
    segment_sampler_batch_size = 2048

    def do_sample(d):
        n = d["notes_np"].shape[0]
        res = []
        for i in range(n):
            seen_game_segments, seen_movement_segments = segment_sampler.sample_for_training(
                d["notes_np"][[i]],
                d["bombs_np"][[i]],
                d["obstacles_np"][[i]],
                d["timestamps"][[i]],
                d["gt_3p_np"][[i]],
                d["lengths"][[i]],
                segment_length,
                1,
                segment_sampler_batch_size,
                stride,
                2.0,
                purview_notes,
                floor_time,
            )
            notes_ = seen_game_segments.notes[:, 2]
            bombs_ = seen_game_segments.bombs[:, 2]
            obstacles_ = seen_game_segments.obstacles[:, 2]
            t_ = seen_game_segments.frames[:, 2]
            my_3p_ = seen_movement_segments.three_p[:, 2:]
            history_ = seen_movement_segments.three_p[:, :2]
            res.append((notes_, bombs_, obstacles_, t_, my_3p_, history_))
        samples = list(reduce(lambda acc, rr: [torch.cat([a, r], dim=0) if isinstance(a, torch.Tensor) else a for a, r in zip(acc, rr)], res))
        return samples

    arrow_files = np.array(sorted(glob.glob(f"{proj_dir}/datasets/boxrr-23/cschell___boxrr-23/default/0.0.0/af25be3ef76b176fbee0a094e82d97a611f9c950/*.arrow")))
    # pbar = tqdm(total=index_pair_sets.shape[0] * index_pair_sets.shape[1])
    pbar = dash(total=index_pair_sets.shape[0] * index_pair_sets.shape[1], global_total=global_total, worker_id=args.job_array_i, progress_dir=f"{proj_dir}/runs/{args.run_name}/{args.out_name}")
    ret_d = {}
    preproc_manifest = []
    for i, (index_pair_set, ty) in enumerate(zip(index_pair_sets, types)):
        cbo_and_3p_for_set = []
        for index_pair in index_pair_set:
            shard_idx, datapoint_idx = index_pair
            shard_file = arrow_files[shard_idx]
            ds = Dataset.from_file(shard_file)
            xror_unpacked = XROR.unpack(ds[int(datapoint_idx)]["xror"])
            beatmap, map_info = open_beatmap_from_unpacked_xror(xror_unpacked)
            left_handed = xror_unpacked.data["info"]["software"]["activity"].get("leftHanded", False)
            cbo_and_3p = load_cbo_and_3p(xror_unpacked, beatmap, map_info, left_handed=left_handed, rescale_yes=True)
            cbo_and_3p_for_set.append(cbo_and_3p)
        d = nanpad_collate_fn([cbo_and_3p_for_set])
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device=device)
        a = do_sample(d)
        id = xror_unpacked.data["info"]["user"]["id"]
        song_hash = xror_unpacked.data["info"]["software"]["activity"]["songHash"]
        difficulty = xror_unpacked.data["info"]["software"]["activity"]["difficulty"]
        song_hash_and_difficulty = f"{song_hash}{difficulty}"

        notes, bombs, obstacles, t, my_3p, history = a
        v = {
            # "id": id,
            "index_pair_set": index_pair_set,
            "notes": notes.detach().cpu(),
            "bombs": bombs.detach().cpu(),
            "obstacles": obstacles.detach().cpu(),
            "t": t.detach().cpu(),
            "my_3p": my_3p.detach().cpu(),
            "history": history.detach().cpu(),
        }
        # first_letters = id[0:2]
        ty = str(ty)
        # ret_d.setdefault(first_letters, {}).setdefault(ty, {}).setdefault(id, [])
        # ret_d[first_letters][ty][id].append(v)

        ret_d.setdefault(ty, {}).setdefault(id, []).append(v)
        # TODO: Also store song hash, difficulty, start time, and end time
        preproc_manifest.append((id, ty, args.job_array_i))

        # Check that the type is correct for this id
        # assert id in umds_df["User ID"].values
        # if ty == "heldout":
        #     assert id in heldout_players["User ID"].values and song_hash_and_difficulty in heldout_maps["Song Hash and Difficulty"].values

        pbar.update(notes.shape[0])
    pbar.close()

    # save ret to outdir/part_{job_array_i:03d}.pt
    df = pd.DataFrame(preproc_manifest, columns=["id", "type", "shard_idx"], index=None)
    df.to_csv(f"{outdir}/{args.job_array_i:04d}.csv", index=False)
    torch.save(ret_d, f"{outdir}/{args.job_array_i:04d}.pkl")

    logger.info(f"Done")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--debug_yes", "-d", action="store_true")  # if set, will pause the program
    parser.add_argument("--job_array_size", type=int, default=1)
    parser.add_argument("--job_array_i", type=int, default=0)
    parser.add_argument("--manifest_dir", type=str, required=True)

    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
