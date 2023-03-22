import transformers
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
        tokenizer=tokenizer, data_path="translation_stack_2048_sample.json", make_prompt=make_prompt, num_proc=1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)