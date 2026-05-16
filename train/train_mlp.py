import os
from argparse import ArgumentParser

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import pauli
from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train import torch_nets
from train.augmenters import PurviewXYAugmenter
from train.torch_nets import MLP
from beaty_common.train_utils import ThroughDataset


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

    run_dir = os.getcwd()
    out_dir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    os.makedirs(out_dir, exist_ok=True)
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")

    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    n_epochs = 50

    current_path = os.getcwd()
    print("Current Path:", current_path)

    # data_path = os.path.join(
    #     proj_dir, "data", "beaterson", "interim", "thats_life_raw_input_output.pkl"
    # )
    d = torch.load(args.data_path, weights_only=False)

    augmenter = PurviewXYAugmenter()
    # augmenter = VanillaAugmenter()
    n_total_steps = int(2.5e6)
    global_steps_elapsed = 0
    epochs_elapsed = 0
    n_steps_per_epoch = 100
    batch_size = 10000
    init_lr = 3e-4
    model = None

    save_every = 1000
    # save_every = 100
    # n_total_steps = int(1e7)
    n_total_steps = int(1e5)
    pbar = tqdm(total=n_total_steps)
    xs_train, ys_train = augmenter.augment(
        # d["song_and_xror_merged"],
        d["song_np"],
        # d["my_pos_sixd"],
        d["my_pos_expm"],
        d["timestamps"],
        batch_size * n_steps_per_epoch,
    )  # peak at just one x-y sample

    # Need to pad xs for MLP input
    dataset = ThroughDataset(xs_train, ys_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    while global_steps_elapsed <= n_total_steps:
        # Model is dynamically defined so I don't have to precompute feature sizes at all
        if model is None:
            logger.info(f"Model hasn't been initialized yet")
            if args.checkpoint_path is None:
                logger.info(f"Initializing from scratch...")
                xs, ys = augmenter.augment(
                    # d["song_and_xror_merged"],
                    d["song_np"],
                    # d["my_pos_sixd"],
                    d["my_pos_expm"],
                    d["timestamps"],
                    1,
                )  # peak at just one x-y sample
                # Need to pad xs for MLP input
                model = MLP(xs.shape[-1], 1024, 1024, ys.shape[-1])
                # model = MLP(xs.shape[-1], xs.shape[-1], 1024, ys.shape[-1])
                model = model.cuda()
                model.rms = model.rms.cuda()
                optimizer = Adam(model.parameters(), init_lr)
                global_steps_elapsed = 0
                epochs_elapsed = 0
            else:
                logger.info(f"Loading from checkpoint at {args.checkpoint_path}...")
                model_dict = torch.load(args.checkpoint_path, weights_only=False)
                model = pauli.load(model_dict)
                model.load_state_dict(model_dict["model_state_dict"])
                model = model.to("cuda")
                optimizer = Adam(model.parameters(), init_lr)
                optimizer.load_state_dict(model_dict["optimizer_state_dict"])
                global_steps_elapsed = model_dict["global_steps_elapsed"]
                epochs_elapsed = model_dict["epochs_elapsed"]
                pbar.update(global_steps_elapsed)
        if epochs_elapsed % save_every == 0 or global_steps_elapsed >= n_total_steps:
            model_d = {
                **pauli.dump(model, torch_nets.__file__),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_steps_elapsed": global_steps_elapsed,
                "epochs_elapsed": epochs_elapsed,
                "args": args,
            }
            save_path = f"{out_dir}/mlp_{epochs_elapsed:06d}.pkl"
            torch.save(model_d, save_path)
            logger.info(f"Saved to {save_path}")

        # Validation step
        logger.info(f"TODO: validation with heldout")
        # with torch.no_grad():
        #     xs_valid = xs_valid.cuda()
        #     ys_valid = ys_valid.cuda()
        #     z, decoded, mu, log_var = model.forward(xs_valid, ys_valid, False)
        #     KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        #     MSE = torch.nn.functional.mse_loss(xs_valid, decoded)
        #     loss = MSE + KLD * args.kld_weight
        #
        #     writer.add_scalar("valid/MSEloss", MSE.item(), global_steps_elapsed)
        #     writer.add_scalar("valid/KLDloss", KLD.item(), global_steps_elapsed)
        #
        #     # for j in range(10):
        #     #     writer.add_scalar(
        #     #         f"valid/mean{j}", torch.mean(z[:, j].abs()), global_steps_elapsed
        #     #     )
        #     #     writer.add_scalar(
        #     #         f"valid/std{j}", torch.std(z[:, j]), global_steps_elapsed
        #     #     )

        # logger.info(
        #     f"Epoch {epochs_elapsed}:\tMSE: {MSE.item():.3f}\tKLD: {KLD.item():.3f}"
        # )

        # lr = init_lr
        lr = init_lr * (1 - (min(1, global_steps_elapsed / n_total_steps)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if global_steps_elapsed < n_total_steps:
            # xs_train, ys_train = augmenter.augment(
            #     d["song_and_xror_merged"],
            #     d["my_pos_sixd"],
            #     d["timestamps"],
            #     batch_size * n_steps_per_epoch,
            # )  # peak at just one x-y sample
            #
            # # Need to pad xs for MLP input
            # dataset = ThroughDataset(xs_train, ys_train)
            # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for i, (x, y) in enumerate(dataloader):
                x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
                y = y.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)

                # noise = torch.randn_like(x) * 0

                # y_hat = model.forward(x + noise, train_yes=True)
                y_hat = model.forward(x, train_yes=True)
                # y_hat = model.forward(x, train_yes=True)
                loss = torch.nn.functional.mse_loss(y, y_hat)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_steps_elapsed += 1

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "steps": global_steps_elapsed})
                if global_steps_elapsed >= n_total_steps:
                    break

            epochs_elapsed += 1
            writer.add_scalar("train/loss", loss.item(), global_steps_elapsed)
            writer.add_scalar("train/lr", lr, global_steps_elapsed)
        else:
            break

    pbar.close()
    writer.flush()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
