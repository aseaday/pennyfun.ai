import transformers
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from pennyfun.datasets.translation_dataset import get_dataset_and_collator
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bigscience/bloomz-1b1")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # optim: str = field(default="adamw_torch")
    # deepspeed='./ds_config.json'
    weight_decay=0.01,
    per_device_train_batch_size = 1
    warmup_steps = 1000
    find_unused_parameters=False
    model_max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1", model_max_length=training_args.model_max_length,
                                              padding_side="right",
                                              use_fast=False,)
    data_module = get_dataset_and_collator(tokenizer)
    model = AutoModelForCausalLM.from_pretrained("./bloomz-1b1", torch_dtype=torch.float16)
    model.half()
    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, **data_module)
    trainer.train()

if __name__ == "__main__":
    # accelerator = Accelerator()
    # accelerator.print(f"{AcceleratorState()}")
    train()
