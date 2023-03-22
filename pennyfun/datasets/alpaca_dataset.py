import transformers
from base import smart_tokenizer, SupervisedDataset, DataCollatorForSupervisedDataset
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def make_prompt(example, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source = prompt_input.format_map(example) if example.get(
        "input", "") != "" else prompt_no_input.format_map(example)
    target = f"{example['output']}{tokenizer.eos_token}"
    return source, target


def get_dataset_and_collator(tokenizer):
    # tokenizer.model_max_length = max_seq_lengh
    smart_tokenizer(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer
    )
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path="./alpaca_data.json", make_prompt=make_prompt, num_proc=40)
    data_collator =  DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


if __name__ == "__main__":
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf",
        padding_side="right",
        use_fast=False,
    )
    r = get_dataset_and_collator(tokenizer)