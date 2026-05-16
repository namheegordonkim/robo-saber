import os
import subprocess
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import transformers
from transformers.integrations import TensorBoardCallback

from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from train.otf_datasets import OTFDatasetMaker, OTFDataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    GenerationConfig,
)


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

    seed = 42
    base_model = "decapoda-research/llama-7b-hf"
    device_map = "auto"

    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = 256
    micro_batch_size = args.micro_batch_size
    num_train_epochs = args.num_train_epochs

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(
        subprocess.check_output(
            "nvidia-smi -L | wc -l", shell=True, encoding="utf-8"
        ).strip()
    )
    print(f"{world_size=}")
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    n_samples_per_epoch = batch_size * 1
    n_samples_per_growth = 1024
    eval_every = 10

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference

    data_path = os.path.join(proj_dir, "data", "beaterson", "interim", "thats_life_raw_input_output.pkl")
    d = torch.load(data_path, weights_only=False)

    dataset_maker = OTFDatasetMaker()
    dataset_maker.setup(d['song_and_xror_merged'], d['my_pos_sixd'])

    dataset = OTFDataset()
    dataset.populate(dataset_maker, d['song_and_xror_merged'], d['my_pos_sixd'], d["timestamps"], 100)
    dataset.dataset

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Initialize the model by loading from a pretrained snapshot and LoRA-fying it.
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # START OVERHAUL

    # Logit production layer overhaul
    # vocab_size = 32000
    # padding_idx = vocab_size
    # model.base_model.embed_tokens = torch.nn.Embedding(vocab_size + 1, 4096, padding_idx=padding_idx, device=model.base_model.embed_tokens.parameters().__next__().device)
    old_head = model.lm_head
    # new_head = torch.nn.Linear(4096, 402, bias=False, device=old_head.parameters().__next__().device)
    new_head = torch.nn.Linear(
        4096,
        dataset_maker.spline_vocab_size + 2,
        bias=False,
        device=old_head.parameters().__next__().device,
    )
    # model.lm_head.weight = torch.nn.Parameter(torch.cat([
    #     old_head.weight,
    #     new_head.weight,
    # ], dim=0))
    # model.lm_head.out_features += 200
    model.lm_head = new_head

    # Token embedding layer overhaul
    old_embed_tokens = model.base_model.embed_tokens
    # new_embed_tokens = torch.nn.Embedding(202, 4096, device=old_embed_tokens.parameters().__next__().device)
    new_embed_tokens = torch.nn.Embedding(
        dataset_maker.spline_vocab_size + dataset_maker.nail_vocab_size + 2,
        4096,
        device=old_embed_tokens.parameters().__next__().device,
    )
    # model.base_model.embed_tokens.weight = torch.nn.Parameter(torch.cat([
    #     old_embed_tokens.weight,
    #     new_embed_tokens.weight,
    # ], dim=0))
    # model.base_model.embed_tokens.num_embeddings += 200
    # conditional_embed_tokens = ConditionalEmbedding(old_embed_tokens, new_embed_tokens)
    # model.base_model.embed_tokens = conditional_embed_tokens
    model.base_model.embed_tokens = new_embed_tokens
    # model.base_model.embed_tokens = torch.nn.Embedding(32000, 4096, padding_idx=31999, device=model.base_model.embed_tokens.parameters().__next__().device)
    # model.lm_head = torch.nn.Linear(4096, 32000, bias=False, device=model.lm_head.parameters().__next__().device)
    # model.config.vocab_size += 200
    # model.config.vocab_size = 202
    model.config.vocab_size = dataset_maker.spline_vocab_size + 2

    # END OVERHAUL

    model = prepare_model_for_kbit_training(model)
    # model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    # model = prepare_model_for_int8_training(model)

    # Training from scratch
    modules_to_save = [
        "embed_tokens",
        "lm_head",
        "rotary_emb",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        # modules_to_save=["lm_head"],
        # modules_to_save=["embed_tokens", "lm_head"],
        modules_to_save=modules_to_save,
        # modules_to_save=["embed_tokens"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # For DDP to output correct
    # for key, _ in model.named_modules():
    #     target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
    #     if target_module_found:
    #         model.get_submodule(key + '.original_module').requires_grad_(False)

    # Note: there are varying number of LoRA layers depending on lora_target_modules
    # Good idea to reset seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    class EmpiricalEvalCallback(TensorBoardCallback):

        def __init__(self):
            super().__init__()
            self.growth_tick = 1

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            to_save = {
                "dataset_maker": dataset_maker,
            }
            torch.save(to_save, os.path.join(args.output_dir, f"dataset_maker.pt"))

        def on_epoch_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if state.epoch > 0:
                # if state.epoch % switcheroo_every == 0:
                #     logger.info("Growing")
                #     kwargs['train_dataloader'].dataset.grow(dataset_maker, s_0_dots_T_train, m_1_dots_T_train, augmenter, n_samples_per_growth)
                #     self.switcheroo_tick += 1

                logger.info("Repopulating")
                train_dataset.populate(
                    tokenizer,
                    dataset_maker,
                    s_0_dots_T_train,
                    m_1_dots_T_train,
                    augmenter,
                    n_samples_per_epoch,
                )

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            # if len(state.log_history) > 0 and state.log_history[-1]['loss'] < 1e-2:
            #     logger.info("Growing")
            #     kwargs['train_dataloader'].dataset.grow(tokenizer, dataset_maker, s_0_dots_T_train, m_1_dots_T_train, augmenter, n_samples_per_growth)
            #     self.growth_tick += 1
            #
            # logger.info(f"{self.growth_tick=}")

            if state.epoch % eval_every != 0:
                return

            with torch.no_grad():

                kwargs["model"].eval()

                # logger.info("Evaluating token accuracy, autoregressive samples")
                # # Training empirical error
                # all_generated = []
                # all_answers = []
                # generation_config = GenerationConfig(pad_token_id=0, eos_token_id=1, bos_token_id=0, max_new_tokens=2)
                #
                # for x in kwargs['train_dataloader']:
                #     input_ids = x['input_ids'][:16]
                #     labels = x['labels'][:16]
                #
                #     tiled_indices = torch.repeat_interleave(torch.arange(labels.shape[-1])[None], labels.shape[0], 0).to(input_ids.device)
                #     # question_lengths = torch.sum(labels == -100, dim=-1) + 0
                #     question_lengths = torch.sum(labels == -100, dim=-1) - 1
                #     yes = torch.logical_and(tiled_indices < question_lengths[:, None], input_ids != 0)
                #     questions = input_ids[yes].reshape(input_ids.shape[0], -1)
                #
                #     yes = tiled_indices >= question_lengths[:, None]
                #     answer_lengths = torch.sum(yes, dim=-1)
                #     receiver_mask = tiled_indices < answer_lengths[:, None]
                #     answers = torch.zeros_like(input_ids)
                #     answers[receiver_mask] = input_ids[yes]
                #
                #     generated = kwargs['model'].generate(inputs=questions, generation_config=generation_config)
                #
                #     all_generated.append(generated[:, -2:])
                #     all_answers.append(answers[:, :2])
                #
                #     # break
                #
                # all_generated = torch.cat(all_generated, dim=0)
                # all_answers = torch.cat(all_answers, dim=0)
                #
                # first_token_empirical_accuracy = torch.mean((all_generated[:, 0] == all_answers[:, 0]).float())
                # logger.info(f"Empirical First Token Accuracy (Train): {first_token_empirical_accuracy.data:.4f}")
                # self.tb_writer.add_scalar("train/first_token_acc", first_token_empirical_accuracy.data, state.global_step)
                #
                # second_token_empirical_accuracy = torch.mean((all_generated[:, 1] == all_answers[:, 1]).float())
                # logger.info(f"Empirical Second Token Accuracy (Train): {second_token_empirical_accuracy.data:.4f}")
                logger.info("Evaluating token accuracy, true samples")
                for token_offset in np.arange(0, 7):
                    # Training empirical error
                    all_generated = []
                    all_answers = []
                    generation_config = GenerationConfig(
                        pad_token_id=0, eos_token_id=1, bos_token_id=0, max_new_tokens=1
                    )
                    cutoff = 47
                    # cutoff = 48
                    for x in kwargs["train_dataloader"]:
                        input_ids = x["input_ids"][:16]
                        labels = x["labels"][:16]

                        tiled_indices = torch.repeat_interleave(
                            torch.arange(labels.shape[-1])[None], labels.shape[0], 0
                        ).to(input_ids.device)
                        # question_lengths = torch.sum(labels == -100, dim=-1) + 1
                        question_lengths = (
                            torch.sum(labels == -100, dim=-1) + token_offset
                        )
                        # question_lengths = torch.sum(labels == -100, dim=-1)
                        # yes = torch.logical_and(tiled_indices < question_lengths[:, None], input_ids != 0)
                        yes = tiled_indices < question_lengths[:, None]
                        # questions = input_ids[yes].reshape(input_ids.shape[0], -1)
                        questions = torch.zeros_like(input_ids)
                        questions[
                            torch.as_tensor(
                                yes.detach().cpu().numpy()[..., ::-1].copy()
                            )
                        ] = input_ids[yes]

                        yes = tiled_indices >= question_lengths[:, None]
                        answer_lengths = torch.sum(yes, dim=-1)
                        receiver_mask = tiled_indices < answer_lengths[:, None]
                        answers = torch.zeros_like(input_ids)
                        answers[receiver_mask] = input_ids[yes]

                        # questions = input_ids[:, :cutoff + token_offset]
                        # answers = input_ids[:, cutoff + token_offset:]

                        generated = kwargs["model"].generate(
                            inputs=questions, generation_config=generation_config
                        )

                        all_generated.append(generated[:, [-1]])
                        all_answers.append(answers[:, [0]])

                        # break

                    all_generated = torch.cat(all_generated, dim=0)
                    all_answers = torch.cat(all_answers, dim=0)

                    token_empirical_acc = torch.mean(
                        (all_generated[:, 0] == all_answers[:, 0]).float()
                    )
                    logger.info(
                        f"Empirical Token {token_offset} Accuracy (Train): {token_empirical_acc.data:.4f}"
                    )
                    self.tb_writer.add_scalar(
                        f"train/token_{token_offset}_acc",
                        token_empirical_acc.data,
                        state.global_step,
                    )

                    # token_abs_diff = torch.mean(torch.abs(all_generated[:, 0] - all_answers[:, 0]).float())
                    # logger.info(f"Empirical Token {token_offset} Difference (Train): {token_abs_diff.data:.4f}")
                    # self.tb_writer.add_scalar(f"train/token_{token_offset}_diff", token_abs_diff.data, state.global_step)

                # # Validation empirical accuracies
                # generation_config = GenerationConfig(pad_token_id=0, eos_token_id=1, bos_token_id=0, max_new_tokens=2)
                # x = valid_dataset[:16]
                # a = torch.as_tensor(np.concatenate(x['input_ids']))
                # b = torch.as_tensor(np.concatenate(x['labels']))
                # input_ids_lengths = np.array([len(s) for s in x['input_ids']])
                # question_lengths = np.array([np.sum(np.array(s) == -100) for s in x['labels']])
                # answer_lengths = np.array([np.sum(np.array(s) != -100) for s in x['labels']])
                # c = int(np.max(question_lengths))
                # d = int(np.max(answer_lengths))
                #
                # tiled_indices = np.arange(c)[None].repeat(16, 0)
                #
                # # input_ids = x['input_ids'][:16]
                # # labels = x['labels'][:16]
                #
                # questions = torch.zeros_like(torch.as_tensor(tiled_indices), dtype=torch.long)
                # receiver_mask = tiled_indices[..., ::-1] < question_lengths[:, None]
                # questions[torch.as_tensor(receiver_mask)] = a[b == -100]
                # questions = questions.to("cuda")
                #
                # tiled_indices = np.arange(d)[None].repeat(16, 0)
                #
                # answers = torch.zeros_like(torch.as_tensor(tiled_indices), dtype=torch.long)
                # receiver_mask = tiled_indices < answer_lengths[:, None]
                # answers[torch.as_tensor(receiver_mask)] = a[b != -100]
                # answers = answers.to("cuda")
                #
                # generated = kwargs['model'].generate(inputs=questions, generation_config=generation_config)
                #
                # first_token_empirical_accuracy = torch.mean((generated[:, -2] == all_answers[:, 0]).float())
                # logger.info(f"Empirical First Token Accuracy (Valid): {first_token_empirical_accuracy.data:.4f}")
                # self.tb_writer.add_scalar("valid/first_token_acc", first_token_empirical_accuracy.data, state.global_step)

                kwargs["model"].train()

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            control.should_save = True
            self.on_save(args, state, control, **kwargs)
            # self.on_epoch_end(args, state, control, **kwargs)

    empirical_eval_callback = EmpiricalEvalCallback()
    output_dir = f"{proj_dir}/runs/{args.run_name}/pi/pi{args.gen:02d}"
    trainer = transformers.Trainer(
        callbacks=[empirical_eval_callback],
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            seed=seed,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # max_grad_norm=1e0,
            max_grad_norm=np.inf,
            num_train_epochs=num_train_epochs,
            learning_rate=3e-4,
            warmup_steps=100,
            # warmup_ratio=0.06,
            lr_scheduler_type="linear",
            logging_dir=f"{proj_dir}/logdir/{args.run_name}/pi/pi{args.gen:02d}",
            fp16=True,
            use_cpu=False,
            logging_steps=1,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.98,
            # weight_decay=1e0,
            weight_decay=1e-2,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            report_to=["tensorboard"],
            run_name=None,
            # gradient_checkpointing=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    with torch.autocast("cuda"):
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
