from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class AccelerateTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self._accelerator = kwargs["accelerator"]
        del kwargs["accelerator"]
        super().__init__(*args, **kwargs)
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataloader = DataLoader(
            self.eval_dataset, shuffle=False, collate_fn=self.data_collator, 
            batch_size=self.args.per_device_eval_batch_size, drop_last=False
        )
        # for step, batch in enumerate(eval_dataloader):
        #     self._accelerator.print(step, batch)
        return self._accelerator.prepare(eval_dataloader)

    def get_train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=self.data_collator, 
            batch_size=self.args.per_device_train_batch_size, drop_last=True
        )
        return self._accelerator.prepare(train_dataloader)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        model = self._accelerator.unwrap_model(self.model)
        state_dict = model.state_dict()
        self._save(output_dir, state_dict=state_dict)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
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
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps * self._accelerator.num_processes)
        self.optimizer = self._accelerator.prepare(self.optimizer)
        self.lr_scheduler = self._accelerator.prepare(self.lr_scheduler)