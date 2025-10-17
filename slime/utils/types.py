from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import torch


@dataclass
class Sample:
    """The sample generated"""

    group_index: Optional[int] = None
    index: Optional[int] = None
    # prompt
    prompt: Union[str, list[dict[str, str]]] = ""
    tokens: list[int] = field(default_factory=list)
    # response
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, dict[str, Any]]] = None
    loss_mask: Optional[list[int]] = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: Optional[list[float]] = None  # Log probabilities from rollout engine

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: Optional[dict] = None

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data["status"] = Sample.Status(data["status"])
        return Sample(**data)

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]


@dataclass
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
