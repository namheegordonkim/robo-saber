import numpy as np
import torch

placeholder_3p_sixd = np.loadtxt("data/placeholder_3p_sixd.txt", dtype=np.float32)


def nanpad_collate_fn(batch: list[list[dict[str, object]] | None]) -> dict[str, object] | None:
    # skip Nones
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # Flatten the batch
    batch = [bb for b in batch for bb in b]

    dd = {}
    for k in list(batch[0].keys()):
        dd[k] = []
        if isinstance(batch[0][k], np.ndarray) or isinstance(batch[0][k], torch.Tensor):
            for d in batch:
                dd[k].append(torch.as_tensor(d[k]))

            # Apply nanpads
            all_lengths = torch.tensor([a.shape[0] for a in dd[k]])
            max_seq_len = all_lengths.max()
            max_seq_len = torch.clip(max_seq_len, min=200)
            lengths_to_go = torch.clip(max_seq_len - all_lengths, min=0)

            nanpads = [torch.ones((lengths_to_go[i], *(dd[k][i].shape[1:]))) * torch.nan for i in range(len(batch))]
            padded_tensors = [torch.cat([a, nanpads[i]], dim=0) for i, a in enumerate(dd[k])]
            stacked_tensors = torch.stack(padded_tensors, dim=0)
            dd[k] = stacked_tensors.to(dtype=torch.float)
        else:
            for d in batch:
                dd[k].append(d[k])
    dd["lengths"] = all_lengths.to(dtype=torch.long)
    return dd
