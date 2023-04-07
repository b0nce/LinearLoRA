from typing import Dict, Any, TypeVar, Mapping
from warnings import warn

import torch
from torch import nn


T_destination = TypeVar("T_destination", bound=Dict[str, Any])


class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, hidden_dim: int = 5) -> None:
        super().__init__()

        assert isinstance(linear, nn.Linear)
        input_size, output_size = linear.in_features, linear.out_features

        self.enc = nn.Linear(
            in_features=input_size, out_features=hidden_dim, bias=False
        )
        self.dec = nn.Linear(
            in_features=hidden_dim, out_features=output_size, bias=False
        )

        self.linear = linear
        assert (
            not linear.weight.requires_grad
        ), "You should freeze the whole model before updating it."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        lora_out = self.dec(self.enc(x))
        return lora_out + orig_out

    def merge(self) -> None:
        new_weight = (
            self.linear.weight.data + self.dec.weight.data @ self.enc.weight.data
        )
        self.linear.weight.data = new_weight

    def state_dict(
        self,
        destination: T_destination = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> None:
        self.merge()
        return self.linear.state_dict(destination, prefix, keep_vars)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> Any:
        warn("You should probably load state dict to the original model!")

        incompatible_keys = self.linear.load_state_dict(state_dict, strict)
        self.enc.weight.data = torch.zeros_like(self.enc.weight.data)
        self.dec.weight.data = torch.zeros_like(self.dec.weight.data)

        return incompatible_keys


def update_model_with_lora(model: nn.Module, rank: int = 5):
    if isinstance(model, LinearLoRA):
        return model

    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearLoRA(module, hidden_dim=rank))
        else:
            updated_module = update_model_with_lora(module, rank=rank)
            setattr(model, name, updated_module)

    return model
