import copy
from dataclasses import dataclass, field
from typing import Dict, Sequence
from loguru import logger

import torch
import transformers
import datasets
from torch.utils.data import Dataset

from pennyfun.utils import jload

IGNORE_INDEX = -100


def smart_tokenizer(special_tokens_dict: Dict,
                    tokenizer: transformers.PreTrainedTokenizer):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)


def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = labels = tokenized.input_ids[0]
    input_ids_lens = labels_lens = input_ids.ne(
        tokenizer.pad_token_id).sum().item()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    source,
    target,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    example = source + target
    example_tokenized, source_tokenized = [_tokenize_fn(
        text, tokenizer) for text in (example, source)]
    input_ids = example_tokenized["input_ids"]
    label = copy.deepcopy(input_ids)
    source_len = source_tokenized["input_ids_lens"]
    label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, label=label)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, make_prompt, num_proc=1):
        super(SupervisedDataset, self).__init__()
        logger.warning("Loading data...")
        dataset = datasets.load_dataset(
            "json", data_files=data_path, num_proc=1)
        self.data = dataset.map(lambda x: preprocess(
            *make_prompt(x, tokenizer), tokenizer), num_proc=num_proc)["train"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.data[i]["input_ids"], labels=self.data[i]["label"])