from __future__ import annotations

import torch


class WarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(0, warmup_epochs)
        self.current_epoch = 0
        for group in self.optimizer.param_groups:
            target_lr = float(group.get("target_lr", group["lr"]))
            initial_lr = float(group.get("warmup_initial_lr", target_lr))
            group["target_lr"] = target_lr
            group["warmup_initial_lr"] = initial_lr
            if self.warmup_epochs > 0:
                group["lr"] = initial_lr

    def step(self) -> None:
        if self.current_epoch < self.warmup_epochs and self.warmup_epochs > 0:
            for group in self.optimizer.param_groups:
                fraction = min(1.0, (self.current_epoch + 1) / self.warmup_epochs)
                group["lr"] = group["warmup_initial_lr"] + (
                    group["target_lr"] - group["warmup_initial_lr"]
                ) * fraction
        self.current_epoch += 1

    def is_warmup_complete(self) -> bool:
        return self.current_epoch >= self.warmup_epochs

    def state_dict(self) -> dict[str, int]:
        return {"warmup_epochs": self.warmup_epochs, "current_epoch": self.current_epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["warmup_epochs"]) != self.warmup_epochs:
            raise ValueError("Warmup epoch count does not match the resume state.")
        self.current_epoch = int(state["current_epoch"])
