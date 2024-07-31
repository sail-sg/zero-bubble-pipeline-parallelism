from dataclasses import dataclass
from typing import List, Optional

from megatron.core.pipeline_parallel.zerobubble.scheduler import ScheduledNode

TYPE_TO_CAT = {
    "F": 0,
    "B": 1,
    "W": 2,
}


@dataclass
class GraphConfig:
    mem_f: List[float] = None
    mem_b: List[float] = None
    mem_w: List[float] = None
    max_mem: Optional[List[float]] = None
    cost_f: List[float] = None
    cost_b: List[float] = None
    cost_w: List[float] = None
    cost_comm: float = 0.0
    print_scaling: int = 1
    max_chunks: int = 1

    def __post_init__(self):
        assert all([isinstance(cost_f, float) for cost_f in self.cost_f])
        assert all([isinstance(cost_b, float) for cost_b in self.cost_b])
        assert all([isinstance(cost_w, float) for cost_w in self.cost_w])
        assert isinstance(self.cost_comm, float)
        assert all([f + b + w == 0 for (f, b, w) in zip(self.mem_f, self.mem_b, self.mem_w)])


class PPGraph:
    def __init__(self, n_stages, n_micro, config: GraphConfig):
        self.n_stages = n_stages
        self.n_micro = n_micro
        self.config = config
        self.fbw_cost = [config.cost_f, config.cost_b, config.cost_w]

    def get_post_validation_time(self, stage, local_order):
        """Get time of POST_VALIDATION"""
        raise NotImplementedError

    def get_cost(self, stage, cat):
        return self.fbw_cost[cat][stage]
