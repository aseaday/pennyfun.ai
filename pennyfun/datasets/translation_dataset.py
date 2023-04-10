import transformers
import loguru
from torch.utils.data import DataLoader
from accelerate import Accelerator
from pennyfun.datasets.base import smart_tokenizer, SupervisedDataset, DataCollatorForSupervisedDataset, DEFAULT_PAD_TOKEN

PROMPT_DICT = {
    "prompt_input": (
        "Translate the following chinese into English.\n"
        "### Input:\n{input}\n\n### Response:\n"
    ),
}


def make_prompt(example, tokenizer):
    source = PROMPT_DICT["prompt_input"].format(input=example["input"])
    target = f"{example['output']}{tokenizer.eos_token}"
    return source, target


def get_dataset_and_collator(tokenizer):
    if tokenizer.pad_token is None:
        smart_tokenizer(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer
        )
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path="translation_stack_512_sample.json", make_prompt=make_prompt, num_proc=1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_train_dataset(tokenizer, accelerator=None):
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path="translation_stack_2000.json", make_prompt=make_prompt, accelerator=accelerator, num_proc=60)
    return train_dataset

def get_eval_dataset(tokenizer, accelerator=None):
    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path="translation_stack_2000_eval.json", make_prompt=make_prompt, accelerator=accelerator, num_proc=60)
    return eval_dataset


def get_accelerate_dataloaders(tokenizer, batch_size=1, accelerator=None):
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path="translation_stack_2000.json", make_prompt=make_prompt, accelerator=accelerator, num_proc=60)
    # def collate_fn(examples):
    #     loguru.logger.info(f"examples: {examples}")
    #     return tokenizer.pad(
    #         examples,
    #         # truncation=True,
    #         padding="longest",
    #         max_length=tokenizer.model_max_length,
    #         return_tensors="pt",
    #     )

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )

    return train_dataloader
