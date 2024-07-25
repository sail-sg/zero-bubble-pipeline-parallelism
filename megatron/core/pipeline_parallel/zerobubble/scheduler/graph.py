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

    def add_comm_node(self, computation_node: ScheduledNode) -> List[ScheduledNode]:
        """Add communication node for a computation node."""
        cat = TYPE_TO_CAT.get(computation_node.type)
        if cat not in (0, 1):  # no communication for W
            return []
        cat_str = "FORWARD" if cat == 0 else "BACKWARD"

        def communicate(send_recv, stage_, comm_direction):
            # noinspection PyTypeChecker
            return ScheduledNode(
                type=send_recv + cat_str,
                chunk=computation_node.chunk,
                stage=stage_,
                minibatch=computation_node.minibatch,
                start_time=computation_node.completion_time,
                completion_time=computation_node.completion_time,  # TODO: consider comm cost in completion time
                comm_direction=comm_direction,
            )

        return self.add_comm_node_impl(computation_node, communicate)

    def add_comm_node_impl(self, computation_node: ScheduledNode, communicate) -> List[ScheduledNode]:
        raise NotImplementedError

    def get_cost(self, stage, cat):
        return self.fbw_cost[cat][stage]
