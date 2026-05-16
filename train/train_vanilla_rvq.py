import os
from argparse import ArgumentParser

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.my_tokenizers import my_kmeans
from train.torch_nets import (
    RunningMeanStd,
)

torch._dynamo.config.optimize_ddp = False


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

    device = torch.device("cuda")

    outdir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    dd = torch.load(
        f"{proj_dir}/data/beaterson/song_3p_preproc/stacked.pkl",
        map_location=device,
    )
    stacked_y = dd["stacked_y"]
    all_lengths = dd["all_lengths"]

    # train_y = stacked_y
    # train_lengths = all_lengths
    train_y = stacked_y[:-1]
    train_lengths = all_lengths[:-1]
    # train_y = stacked_y[:1]
    # train_lengths = all_lengths[:1]

    # train_lengths[:] = 1000

    valid_y = stacked_y[[-1]]
    valid_lengths = all_lengths[[-1]]

    npr_st = np.random.get_state()
    np.random.seed(0)

    batch_size = int(1e5)
    sample_size = 1024
    segment_length = args.segment_length
    seen_idxs = (torch.rand(sample_size) * train_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    seen_t_start = (
        torch.rand(sample_size, device=device)
        * (train_lengths[seen_idxs] - segment_length)
    ).to(dtype=torch.long)
    segment_ts = torch.arange(segment_length, device=device) + seen_t_start[:, None]
    seen_segment_ys = torch.take_along_dim(
        train_y[seen_idxs], segment_ts[:, :, None], dim=1
    )

    unseen_idxs = (torch.rand(sample_size) * valid_lengths.shape[0]).to(
        dtype=torch.long, device=device
    )
    unseen_t_start = (
        torch.rand(sample_size, device=device)
        * (valid_lengths[unseen_idxs] - segment_length)
    ).to(dtype=torch.long)
    segment_ts = torch.arange(segment_length, device=device) + unseen_t_start[:, None]
    unseen_segment_ys = torch.take_along_dim(
        valid_y[unseen_idxs], segment_ts[:, :, None], dim=1
    )

    np.random.set_state(npr_st)

    # Set up RVQ stuff
    train_y.to(device)
    rms = RunningMeanStd(shape=(train_y.shape[-1]))
    rms.to(device)
    yy = train_y.reshape(-1, train_y.shape[-1])
    nan_no = ~torch.isnan(yy).any(-1)
    yy = yy[nan_no]
    rms.update(yy)
    random_idxs = torch.randperm(yy.shape[0])[:batch_size]
    yy = yy[random_idxs]
    for sentence_length in range(1, 11):

        codebooks = torch.zeros(
            sentence_length, args.vocab_size, train_y.shape[-1], device=device
        )
        yy = rms.normalize(yy)
        with torch.no_grad():
            # RVQ initialization
            quantized = torch.zeros_like(yy)
            for i in range(sentence_length):
                means, assignments = my_kmeans(
                    yy - quantized, args.vocab_size, 50, None, False
                )
                codebooks.data[i] = means
                quantized += codebooks[i][assignments]
            quantized = rms.unnormalize(quantized)
            yy = rms.unnormalize(yy)
            q_err = F.mse_loss(quantized, yy).item()
            logger.info(f"Sentence length: {sentence_length} Initial quantization error: {q_err}")

            for j in range(2):
                # quantize seen segments
                if j == 0:
                    seen_yy = seen_segment_ys.reshape(-1, seen_segment_ys.shape[-1])
                else:
                    seen_yy = unseen_segment_ys.reshape(-1, seen_segment_ys.shape[-1])

                seen_yy = rms.normalize(seen_yy)
                quantized = torch.zeros_like(seen_yy)
                for i in range(sentence_length):
                    resid = seen_yy - quantized
                    deltas = resid[:, None] - codebooks[i][None]
                    k = torch.argmin(torch.mean(deltas ** 2, dim=-1), dim=-1)
                    quantized += codebooks[i][k]
                quantized = rms.unnormalize(quantized)
                seen_yy = rms.unnormalize(seen_yy)
                q_err = F.mse_loss(quantized, seen_yy).item()

                if j == 0:
                    logger.info(f"Sentence length: {sentence_length} Seen quantization error: {q_err}")
                else:
                    logger.info(f"Sentence length: {sentence_length} Unseen quantization error: {q_err}")

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--sentence_length", type=int, default=1)
    parser.add_argument("--segment_length", type=int, default=1)
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--cached_yes", action="store_true")
    args = parser.parse_args()

    main()
