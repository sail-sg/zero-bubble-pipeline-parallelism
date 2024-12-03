from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler import group_interleaved_1f1b, basic1f1b, vpp
from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import validate_communication, \
    CommSet, add_communication_nodes, reorder_communication, \
    add_communication_nodes_without_sorting, add_post_validation_nodes, tag_rollback_communication
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode
from megatron.core.pipeline_parallel.zerobubble.scheduler.offloading import add_offload
from megatron.core.pipeline_parallel.zerobubble.scheduler.passes import pre_validate, add_send_recv_peer_stage, \
    add_time, print_schedule


def old_run_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    offload_time=None,
    post_validation=False,
    validate=True,
) -> List[List[ScheduledNode]]:
    pre_validate(local_order)
    local_order = add_send_recv_peer_stage(config, local_order)
    local_order = add_time(config, local_order)
    local_order = add_offload(config, local_order, offload_time)
    local_order = old_run_communication_passes(config, local_order, post_validation)
    print_schedule(local_order)
    if validate:
        validate_communication(local_order)
    return local_order


def old_run_communication_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    post_validation: bool,
) -> List[List[ScheduledNode]]:
    comm_set = CommSet()
    if post_validation:
        local_order = add_post_validation_nodes(config, comm_set, local_order)
    local_order = add_communication_nodes(config, comm_set, local_order)
    local_order = reorder_communication(config, comm_set, local_order)
    if post_validation:
        local_order = tag_rollback_communication(config, local_order)
    return local_order


def new_run_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    offload_time=None,
    post_validation=False,
    validate=True,
) -> List[List[ScheduledNode]]:
    pre_validate(local_order)
    local_order = add_send_recv_peer_stage(config, local_order)
    local_order = add_time(config, local_order)
    local_order = add_offload(config, local_order, offload_time)
    local_order = new_run_communication_passes(config, local_order, post_validation)
    print_schedule(local_order)
    if validate:
        validate_communication(local_order, debug=True)
    return local_order


def new_run_communication_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    post_validation: bool,
) -> List[List[ScheduledNode]]:
    # if get_args().enable_optimizer_post_validation:
    #     local_order = add_post_validation_nodes(config, comm_set, local_order)
    local_order = add_communication_nodes_without_sorting(config, local_order)
    # if get_args().enable_optimizer_post_validation:
        # local_order = add_post_validation_nodes_before_deadline(config, comm_set, local_order)
        # local_order = tag_rollback_communication(config, local_order)
    return local_order


def create_dummy_config():
    return GraphConfig.basic_config(
        f=1000.0,
        b=1000.0,
        w=1000.0,
        n_stages=4,
        n_micro=16,
        max_chunks=1,
    )


def test_new_comm_impl_on_1f1b():
    config = create_dummy_config()
    local_order = basic1f1b.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, local_order)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, local_order)

    assert len(old) == len(new)
    stage = 0
    for old_stage_nodes, new_stage_nodes in zip(old, new):
        assert len(old_stage_nodes) == len(new_stage_nodes)
        for o, n in zip(old_stage_nodes, new_stage_nodes):
            assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
        stage += 1


def test_new_comm_impl_on_vpp():
    config = create_dummy_config()
    local_order = vpp.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, local_order)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, local_order)

    assert len(old) == len(new)
    stage = 0
    for old_stage_nodes, new_stage_nodes in zip(old, new):
        assert len(old_stage_nodes) == len(new_stage_nodes)
        for o, n in zip(old_stage_nodes, new_stage_nodes):
            assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
        stage += 1
