from dataclasses import dataclass
from typing import List


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
    max_mem: List[float] = None
    cost_f: List[int] = None
    cost_b: List[int] = None
    cost_w: List[int] = None
    cost_comm: int = 0
    print_scaling: int = 1
    max_chunks: int = 1

    def __post_init__(self):
        assert all([type(cost_f) is int for cost_f in self.cost_f])
        assert all([type(cost_b) is int for cost_b in self.cost_b])
        assert all([type(cost_w) is int for cost_w in self.cost_w])
        assert type(self.cost_comm) is int
        assert all([f + b + w == 0 for (f, b, w) in zip(self.mem_f, self.mem_b, self.mem_w)])


class PPGraph:
    def __init__(self, n_stages, n_micro, config: GraphConfig):
        self.n_stages = n_stages
        self.n_micro = n_micro
        self.config = config
        self.fbw_cost = [config.cost_f, config.cost_b, config.cost_w]

    def get_n_node(self):
        """Get number of nodes in total"""
        raise NotImplementedError

    def get_id(self, cat, chunk, stage, micro):
        """Get node id"""
        raise NotImplementedError

    def get_post_validation_time(self, stage, local_order):
        """Get time of POST_VALIDATION"""
        raise NotImplementedError

    def get_cost(self, stage, cat):
        return self.fbw_cost[cat][stage]
