"""Fixed structured advice for controlled experiments, not learned advice validity."""
from dataclasses import dataclass


@dataclass
class AdviceGate:
    mode: str
    decay_steps: float = 100.0
    status: str = "Pending"
    switches: int = 0

    def __post_init__(self):
        if self.mode not in {"none", "persistent", "decay", "context", "direct", "weak"}:
            raise ValueError(f"Unknown feedback mode: {self.mode}")

    def weight(self, valid, condition, age, completed=False, revoked=False):
        import math
        previous = self.status
        if revoked or self.status == "Revoked":
            self.status = "Revoked"
        elif completed or self.status == "Completed":
            self.status = "Completed"
        elif valid and condition:
            self.status = "Active"
        elif previous in {"Active", "Suspended"}:
            self.status = "Suspended"
        else:
            self.status = "Pending"
        self.switches += int(self.status != previous)
        if self.mode == "none" or not valid or self.status in {"Completed", "Revoked"}:
            return 0.0
        if self.mode == "persistent":
            return 1.0
        if self.mode == "weak":
            return 0.25
        if self.mode == "decay":
            return math.exp(-age / self.decay_steps)
        if self.mode == "direct":
            return float(condition)
        return float(self.status == "Active")
