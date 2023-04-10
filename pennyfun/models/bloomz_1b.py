import transformers
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from typing import Optional
import json
import loguru
from transformers import AutoTokenizer, AutoModelForCausalLM
from pennyfun.datasets.translation_dataset import get_train_dataset,get_eval_dataset
from pennyfun.trainers.accelerateTrainer import AccelerateTrainer
from torch.utils.data import Dataset
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
    eval_steps: int = field(default=100)
    num_train_epochs: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=3e-4)
    per_device_train_batch_size: int = field(default=1) 
    per_device_eval_batch_size: int = field(default=2)
    warmup_ratio: float = field(default=0.1)
    # warmup_steps: int = field(default=2000)
    find_unused_parameters: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    save_steps: int = field(default=100)
    # fp16: bool = field(default=True)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
def train():
    accelerator = Accelerator()
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1", model_max_length=training_args.model_max_length,
                                              use_fast=False,)
    with accelerator.main_process_first():
        train_dataset = get_train_dataset(tokenizer, accelerator)
    with accelerator.main_process_first():
        eval_dataset = get_eval_dataset(tokenizer, accelerator)
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b1")
    model = accelerator.prepare(model)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=training_args.learning_rate)
    # lr = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps)
    # optimizer, lr = accelerator.prepare(optimizer, lr)
    trainer = AccelerateTrainer(model, tokenizer=tokenizer, args=training_args,
                                train_dataset=train_dataset, eval_dataset=eval_dataset, accelerator=accelerator, data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    trainer.save_model("bloom-translation")

if __name__ == "__main__":
    # accelerator = Accelerator()
    # accelerator.print(f"{AcceleratorState()}")
    train()
