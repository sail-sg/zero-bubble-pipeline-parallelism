from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    chunk: int
    stage: int
    minibatch: int
    start_time: int
    completion_time: int
    rollback: bool = False
