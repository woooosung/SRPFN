import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

try:
    from encoder import Encoder, MLPEncoder
except Exception:
    Encoder = None

    class MLPEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, **kwargs):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim * 2),
                nn.GELU(),
                nn.Linear(output_dim * 2, output_dim),
            )

        def forward(self, interaction_vectors, **kwargs):
            return self.net(interaction_vectors)


def get_encoder_generator():
    if Encoder is not None:
        return Encoder
    return MLPEncoder

def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5, 
    eta_min_ratio=0.1,
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def ema_update_(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    model_state = model.state_dict()
    for key, ema_value in ema_model.state_dict().items():
        value = model_state[key]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(value, alpha=(1.0 - decay))
        else:
            ema_value.copy_(value)


def get_ema_decay(step: int, base_decay: float = 0.999, warmup_steps: int = 0):
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_decay

    start = 0.90
    t = step / max(1, warmup_steps)
    return start + (base_decay - start) * t
