from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CommDirection(Enum):
    NEXT = 0
    PREV = 1


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    stage: int
    minibatch: int
    start_time: int
    completion_time: int
    chunk: int = 0
    comm_direction: Optional[CommDirection] = None
    rollback: bool = False
