from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CommDirection(Enum):
    NEXT = 0
    PREV = 1


@dataclass(eq=True, frozen=True)
class NodeKey:
    type: str
    stage: int
    minibatch: int
    chunk: int = 0

    def __hash__(self):
        return hash((self.type, self.stage, self.minibatch, self.chunk))


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    stage: int
    minibatch: int
    chunk: int = 0
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    prev_compute_node: Optional[NodeKey] = None
    # Only for computation node
    recv_peer_stage: Optional[int] = None
    send_peer_stage: Optional[int] = None
    # Only for communication node
    comm_direction: Optional[CommDirection] = None
    rollback: bool = False

    def get_key(self):
        return NodeKey(self.type, self.stage, self.minibatch, self.chunk)
