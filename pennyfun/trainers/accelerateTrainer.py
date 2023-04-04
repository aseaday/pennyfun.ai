from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader


class AccelerateTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self._accelerator = kwargs["accelerator"]
        del kwargs["accelerator"]
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=self.data_collator, 
            batch_size=self.args.per_device_train_batch_size, drop_last=True
        )
        return self._accelerator.prepare(train_dataloader)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        self._accelerator.backward(loss)
        return loss.detach()
