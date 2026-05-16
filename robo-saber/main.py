from robosaber.checkpoint import ensure_checkpoint


def main() -> None:
    ckpt = ensure_checkpoint()
    print(f"Checkpoint ready at: {ckpt}")
    # TODO: load model + run inference here.
