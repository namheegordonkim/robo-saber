import numpy as np
import torch
from datasets import Dataset
from torch.utils.data.dataset import T_co

from train.augmenters import PurviewXYAugmenter
from train.my_tokenizers import BucketizeTokenizer
from beaty_common.train_utils import RunningMeanStd


class OTFMLPDatasetMaker:
    """
    Given Numpy arrays, prepares HuggingFace training / validation data
    """

    def __init__(self):
        # To be dynamically populated during `setup()`
        self.x_tokenizer = None
        self.x_scaler = None

        self.x_vocab_size = 2000
        self.x_code_length = None

        self.y_tokenizer = None
        self.y_scaler = None

        self.y_vocab_size = 2000
        self.y_code_length = None

    def setup(self, setup_x: np.ndarray, setup_y: np.ndarray):
        self.x_scaler = RunningMeanStd(shape=(setup_x.shape[1:]))
        self.y_scaler = RunningMeanStd(shape=(setup_y.shape[1:]))

        self.x_scaler.update(setup_x)
        self.y_scaler.update(setup_y)

        self.x_tokenizer = BucketizeTokenizer(
            setup_x.shape[-1], self.x_vocab_size, -3, 3
        )
        self.y_tokenizer = BucketizeTokenizer(
            setup_y.shape[-1], self.y_vocab_size, -3, 3
        )

    def make(
        self,
        augmenter: PurviewXYAugmenter,
        source_x: np.ndarray,
        source_y: np.ndarray,
        timestamps: np.ndarray,
        size: int,
    ) -> Dataset:
        print("huehue")

        aug_x, aug_y = augmenter.augment(source_x, timestamps, size)

        purview_sec = 2
        start_times = np.random.uniform(
            timestamps.min(), timestamps.max() - purview_sec, size=size
        )
        end_times = start_times + purview_sec

        x_within_purview_yes = np.logical_and(
            start_times[None] <= source_x[:, [0]], end_times[None] >= source_x[:, [0]]
        ).T
        y_within_purview_yes = np.logical_and(
            start_times[None] <= timestamps[:, None],
            end_times[None] >= timestamps[:, None],
        ).T

        with torch.no_grad():
            x_tensor = torch.clip(
                torch.as_tensor(
                    self.x_scaler.normalize(source_x), dtype=torch.float, device="cuda"
                ),
                -3.0,
                3.0,
            )
            y_tensor = torch.clip(
                torch.as_tensor(
                    self.y_scaler.normalize(source_y), dtype=torch.float, device="cuda"
                ),
                -3.0,
                3.0,
            )

        x_encoded, x_quantized = self.x_tokenizer.encode(x_tensor)
        y_encoded, y_quantized = self.y_tokenizer.encode(y_tensor)

        # x_within_purview = source_x[x_within_purview_yes]
        # y_within_purview = source_y[y_within_purview_yes]

        # Get quantization error
        q_deltas = torch.abs(x_tensor - x_quantized).detach().cpu().numpy()
        q_mean = q_deltas.mean()
        q_std = q_deltas.std()
        q_max = q_deltas.max()

        print(f"Mean x quantization error : {q_mean:.3f}")
        print(f"Std x quantization error : {q_std:.3f}")
        print(f"Max x quantization error : {q_max:.3f}")

        # Get quantization error
        q_deltas = torch.abs(y_tensor - y_quantized).detach().cpu().numpy()
        q_mean = q_deltas.mean()
        q_std = q_deltas.std()
        q_max = q_deltas.max()

        print(f"Mean y quantization error : {q_mean:.3f}")
        print(f"Std y quantization error : {q_std:.3f}")
        print(f"Max y quantization error : {q_max:.3f}")

        # + 12 and + 2 because:
        # 12 total output vocabulary; 0 for padding, 1 for <EOS>, and 2-11 for the 10 digits (0-9)
        # To make the input vocabulary non-overlapping with the output vocabulary, we start input vocabulary at 12
        input_ids_np = np.concatenate(
            [
                x_patches_encoded.cpu()
                .detach()
                .numpy()
                .reshape(x_patches_encoded.shape[0], -1)
                + 12,
                source_y[:, None] + 2,
                np.ones((x_patches_encoded.shape[0], 1), dtype=int),
            ],
            axis=-1,
        )
        labels_np = input_ids_np * 1
        labels_np[:, :-2] = -100  # "unused" for HuggingFace Llama
        attention_mask_np = np.ones_like(input_ids_np)

        input_ids = input_ids_np.tolist()
        labels = labels_np.tolist()
        attention_mask = attention_mask_np.tolist()

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        dataset = Dataset.from_dict(data_dict)
        return dataset


class OTFDatasetMaker:
    """
    Given Numpy arrays, prepares HuggingFace training / validation data
    """

    def __init__(self):
        # To be dynamically populated during `setup()`
        self.x_tokenizer = None
        self.x_scaler = None

        self.x_vocab_size = 2000
        self.x_code_length = None

        self.y_tokenizer = None
        self.y_scaler = None

        self.y_vocab_size = 2000
        self.y_code_length = None

    def setup(self, setup_x: np.ndarray, setup_y: np.ndarray):
        self.x_scaler = RunningMeanStd(shape=(setup_x.shape[1:]))
        self.y_scaler = RunningMeanStd(shape=(setup_y.shape[1:]))

        self.x_scaler.update(setup_x)
        self.y_scaler.update(setup_y)

        self.x_tokenizer = BucketizeTokenizer(
            setup_x.shape[-1], self.x_vocab_size, -3, 3
        )
        self.y_tokenizer = BucketizeTokenizer(
            setup_y.shape[-1], self.y_vocab_size, -3, 3
        )

    def make(
        self,
        augmenter: PurviewXYAugmenter,
        source_x: np.ndarray,
        source_y: np.ndarray,
        timestamps: np.ndarray,
        size: int,
    ) -> Dataset:
        print("huehue")

        aug_x, aug_y = augmenter.augment(source_x, timestamps, size)

        purview_sec = 2
        start_times = np.random.uniform(
            timestamps.min(), timestamps.max() - purview_sec, size=size
        )
        end_times = start_times + purview_sec

        x_within_purview_yes = np.logical_and(
            start_times[None] <= source_x[:, [0]], end_times[None] >= source_x[:, [0]]
        ).T
        y_within_purview_yes = np.logical_and(
            start_times[None] <= timestamps[:, None],
            end_times[None] >= timestamps[:, None],
        ).T

        with torch.no_grad():
            x_tensor = torch.clip(
                torch.as_tensor(
                    self.x_scaler.normalize(source_x), dtype=torch.float, device="cuda"
                ),
                -3.0,
                3.0,
            )
            y_tensor = torch.clip(
                torch.as_tensor(
                    self.y_scaler.normalize(source_y), dtype=torch.float, device="cuda"
                ),
                -3.0,
                3.0,
            )

        x_encoded, x_quantized = self.x_tokenizer.encode(x_tensor)
        y_encoded, y_quantized = self.y_tokenizer.encode(y_tensor)

        # x_within_purview = source_x[x_within_purview_yes]
        # y_within_purview = source_y[y_within_purview_yes]

        # Get quantization error
        q_deltas = torch.abs(x_tensor - x_quantized).detach().cpu().numpy()
        q_mean = q_deltas.mean()
        q_std = q_deltas.std()
        q_max = q_deltas.max()

        print(f"Mean x quantization error : {q_mean:.3f}")
        print(f"Std x quantization error : {q_std:.3f}")
        print(f"Max x quantization error : {q_max:.3f}")

        # Get quantization error
        q_deltas = torch.abs(y_tensor - y_quantized).detach().cpu().numpy()
        q_mean = q_deltas.mean()
        q_std = q_deltas.std()
        q_max = q_deltas.max()

        print(f"Mean y quantization error : {q_mean:.3f}")
        print(f"Std y quantization error : {q_std:.3f}")
        print(f"Max y quantization error : {q_max:.3f}")

        # + 12 and + 2 because:
        # 12 total output vocabulary; 0 for padding, 1 for <EOS>, and 2-11 for the 10 digits (0-9)
        # To make the input vocabulary non-overlapping with the output vocabulary, we start input vocabulary at 12
        input_ids_np = np.concatenate(
            [
                x_patches_encoded.cpu()
                .detach()
                .numpy()
                .reshape(x_patches_encoded.shape[0], -1)
                + 12,
                source_y[:, None] + 2,
                np.ones((x_patches_encoded.shape[0], 1), dtype=int),
            ],
            axis=-1,
        )
        labels_np = input_ids_np * 1
        labels_np[:, :-2] = -100  # "unused" for HuggingFace Llama
        attention_mask_np = np.ones_like(input_ids_np)

        input_ids = input_ids_np.tolist()
        labels = labels_np.tolist()
        attention_mask = attention_mask_np.tolist()

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        dataset = Dataset.from_dict(data_dict)
        return dataset


class OTFDataset(torch.utils.data.Dataset):

    def __getitem__(self, index) -> T_co:
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()

    def __init__(self):
        self.dataset = None

    def populate(
        self,
        dataset_maker: OTFDatasetMaker,
        source_x,
        source_y,
        timestamps,
        size: int,
    ):
        augmenter = PurviewXYAugmenter()
        self.dataset = dataset_maker.make(
            augmenter, source_x, source_y, timestamps, size
        )
