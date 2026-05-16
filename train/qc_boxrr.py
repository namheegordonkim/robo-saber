import glob
import multiprocessing
import os
import traceback
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from beaty_common.bsmg_xror_utils import load_cbo_and_3p
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from xror.xror import XROR

warnings.filterwarnings("ignore")

device = torch.device("cuda")


def main():
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12346,
            stdout_to_server=True,
            stderr_to_server=True,
            suspend=False,
        )
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        job_array_i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        job_array_i = 0
    print(f"Job array index is {job_array_i} out of {args.job_array_size - 1}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    outdir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    in_arrow_dir = os.path.abspath(args.in_boxrr_dir)

    arrow_files = np.array(sorted(glob.glob(f"{proj_dir}/datasets/boxrr-23/cschell___boxrr-23/default/0.0.0/af25be3ef76b176fbee0a094e82d97a611f9c950/*.arrow")))
    shard_idxs = np.arange(len(arrow_files))
    shard_names = sorted(glob.glob(f"{in_arrow_dir}/cschell___boxrr-23/default/0.0.0/**/*.arrow", recursive=True))
    shard_sizes = np.loadtxt(f"{in_arrow_dir}/boxrr_shard_sizes.txt").astype(int)
    if args.job_array_size > 0:
        arrow_files = np.array_split(arrow_files, args.job_array_size)[job_array_i]
        shard_idxs = np.array_split(shard_idxs, args.job_array_size)[job_array_i]
        shard_sizes = np.array_split(shard_sizes, args.job_array_size)[job_array_i]

    pbar = tqdm(total=shard_sizes.sum())
    valids = []
    pool = multiprocessing.Pool(1)

    for shard_idx, arrow_file, shard_size in zip(shard_idxs, arrow_files, shard_sizes):
        shard_ds = Dataset.from_file(arrow_file)

        for idx in range(shard_size):
            try:
                xror_raw = shard_ds[idx]["xror"]
                future = pool.apply_async(XROR.unpack, args=(xror_raw,))
                xror = future.get(10)
                if xror.data['info']['activity']['mode'] != "Standard":
                    continue
                d = load_cbo_and_3p(xror)
                if d["timestamps"].shape[0] > 1000 and np.isnan(d["gt_3p_np"]).sum() == 0 and np.isinf(d["gt_3p_np"]).sum() == 0:
                    valids.append(np.array([shard_idx, idx, xror.data['info']['user']['totalScore']]))
            except Exception as e:
                print("Exception occurred")
                print(traceback.format_exc())
            pbar.update(1)

        np.savetxt(f"{outdir}/valids_{job_array_i:03d}.txt", np.array(valids), fmt="%d")

    pbar.close()
    valids = np.array(valids)
    # np.savetxt(f"{outdir}/valids_{job_array_i:03d}.txt", valids, fmt="%d")

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--in_boxrr_dir", type=str, required=True)
    parser.add_argument("--job_array_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main()
