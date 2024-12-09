import copy
from typing import List

import pytest

from megatron.core.pipeline_parallel.zerobubble.scheduler import basic1f1b, vpp, v1f1b, zb, zbv, group_interleaved_1f1b, \
    zbv_greedy
from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import validate_communication, \
    CommSet, add_communication_nodes, reorder_communication, \
    add_communication_nodes_without_sorting, add_post_validation_nodes, tag_rollback_communication
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode
from megatron.core.pipeline_parallel.zerobubble.scheduler.offloading import add_offload, smooth_start_time
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
    if offload_time:
        local_order = add_offload(config, local_order, offload_time)
        local_order = smooth_start_time(local_order)
    local_order = old_run_communication_passes(config, local_order, post_validation)
    print_schedule(local_order)
    if validate:
        validate_communication(local_order, debug=False)
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
    if offload_time:
        local_order = add_offload(config, local_order, offload_time)
    local_order = add_communication_nodes_without_sorting(config, local_order, post_validation)
    print_schedule(local_order)
    if validate:
        validate_communication(local_order, debug=False)
    return local_order


def create_dummy_config(n_stages=4, n_micro=16, max_chunks=1):
    config = GraphConfig.basic_config(
        f=1.0,
        b=1.0,
        w=1.0,
        n_stages=n_stages,
        n_micro=n_micro,
        max_chunks=max_chunks,
    )
    config.mem_f = [1000.0] * n_stages
    config.mem_b = [1000.0] * n_stages
    config.mem_w = [1000.0] * n_stages
    return config


TEST_SETTINGS = [
    (4, 4),
    (4, 8),
    (4, 16),
    (2, 2),
    (2, 4),
    (8, 4),
    (8, 8),
    (8, 16),
]


VPP_TEST_SETTINGS = [
    (2, 2),
    (2, 4),
    (4, 4),
    (4, 8),
    (4, 16),
    (8, 8),
    (8, 16),
]


@pytest.mark.parametrize("n_stages,n_micro", TEST_SETTINGS)
def test_new_comm_impl_on_1f1b(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro)
    local_order = basic1f1b.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order))
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order))

    assert len(old) == len(new)
    stage = 0
    for old_stage_nodes, new_stage_nodes in zip(old, new):
        assert len(old_stage_nodes) == len(new_stage_nodes)
        for o, n in zip(old_stage_nodes, new_stage_nodes):
            assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
        stage += 1


@pytest.mark.parametrize("n_stages,n_micro", VPP_TEST_SETTINGS)
def test_new_comm_impl_on_vpp(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro)
    local_order = vpp.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order))
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order))

    assert len(old) == len(new)
    stage = 0
    for old_stage_nodes, new_stage_nodes in zip(old, new):
        assert len(old_stage_nodes) == len(new_stage_nodes)
        for o, n in zip(old_stage_nodes, new_stage_nodes):
            assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
        stage += 1


@pytest.mark.parametrize("n_stages,n_micro", TEST_SETTINGS)
def test_new_comm_impl_on_1f1bv(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro, max_chunks=2)
    local_order = v1f1b.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order))
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order))

    assert len(old) == len(new)
    stage = 0
    for old_stage_nodes, new_stage_nodes in zip(old, new):
        assert len(old_stage_nodes) == len(new_stage_nodes)
        for o, n in zip(old_stage_nodes, new_stage_nodes):
            assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
        stage += 1


OFFLOAD_TEST_SETTINGS = [
    # (2, 2),
    (2, 4),
    (4, 4),
    (4, 8),
    (4, 16),
    (8, 4),
    (8, 8),
    (8, 16),
]


def create_offload_dummy_config(n_stages=4, n_micro=16, max_chunks=1):
    config = GraphConfig.basic_config(
        f=1.0,
        b=1.0,
        w=1.0,
        n_stages=n_stages,
        n_micro=n_micro,
        max_chunks=max_chunks,
    )
    return config


@pytest.mark.parametrize("n_stages,n_micro", OFFLOAD_TEST_SETTINGS)
def test_new_comm_impl_with_offload(n_stages, n_micro):
    config = create_offload_dummy_config(n_stages=n_stages, n_micro=n_micro, max_chunks=2)
    print(f"offload config: {config}")
    local_order = group_interleaved_1f1b.create_schedule(
        config,
        cpu_offload=True,
        recompute_granularity=None,
        recompute_method="chunk",
        recompute_num_layers=1,
        interleave_group_size=4,
        offload_chunk_num=1,
    )
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order), offload_time=0.5)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order), offload_time=0.5)

    assert len(old) == len(new)
    # Not exactly the same as before
    # stage = 0
    # for old_stage_nodes, new_stage_nodes in zip(old, new):
    #     assert len(old_stage_nodes) == len(new_stage_nodes)
    #     for o, n in zip(old_stage_nodes, new_stage_nodes):
    #         assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
    #     stage += 1


@pytest.mark.parametrize("n_stages,n_micro", TEST_SETTINGS)
def test_new_comm_impl_on_zb(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro)
    local_order = zb.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order), post_validation=True, validate=False)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order), post_validation=True)

    assert len(old) == len(new)
    # stage = 0
    # for old_stage_nodes, new_stage_nodes in zip(old, new):
    #     assert len(old_stage_nodes) == len(new_stage_nodes)
    #     for o, n in zip(old_stage_nodes, new_stage_nodes):
    #         assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
    #     stage += 1


@pytest.mark.parametrize("n_stages,n_micro", TEST_SETTINGS)
def test_new_comm_impl_on_zbv(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro, max_chunks=2)
    f_mid, b_mid, w_mid, c = 1000.0, 1000.0, 1000.0, 10.0

    hidden_size = 4096
    num_attention_heads = 8
    seq_length = 4096
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx
    pp_graph = zbv.PipelineGraph(
        config.n_stages,
        config.n_micro,
        f_mid, b_mid, w_mid, c,
        f_mem=f_mem_approx, b_mem=b_mem_approx, w_mem=w_mem_approx,
        max_mem=None
    )
    local_order = pp_graph.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order), post_validation=True, validate=False)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order), post_validation=True)
    assert len(old) == len(new)
    # stage = 0
    # for old_stage_nodes, new_stage_nodes in zip(old, new):
    #     assert len(old_stage_nodes) == len(new_stage_nodes)
    #     for o, n in zip(old_stage_nodes, new_stage_nodes):
    #         assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
    #     stage += 1


@pytest.mark.parametrize("n_stages,n_micro", TEST_SETTINGS)
def test_new_comm_impl_on_zbv_min(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro, max_chunks=2)
    f_mid, b_mid, w_mid, c = 1000.0, 1000.0, 1000.0, 10.0
    pp_graph = zbv_greedy.PipelineGraph(
        config.n_stages,
        config.n_micro,
        "min",
        f_mid, b_mid, w_mid, c
    )
    local_order = pp_graph.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order), post_validation=True, validate=False)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order), post_validation=True)
    assert len(old) == len(new)
    # stage = 0
    # for old_stage_nodes, new_stage_nodes in zip(old, new):
    #     assert len(old_stage_nodes) == len(new_stage_nodes)
    #     for o, n in zip(old_stage_nodes, new_stage_nodes):
    #         assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
    #     stage += 1


@pytest.mark.parametrize("n_stages,n_micro", TEST_SETTINGS)
def test_new_comm_impl_on_zbv_half(n_stages, n_micro):
    config = create_dummy_config(n_stages=n_stages, n_micro=n_micro, max_chunks=2)
    f_mid, b_mid, w_mid, c = 1000.0, 1000.0, 1000.0, 10.0
    pp_graph = zbv_greedy.PipelineGraph(
        config.n_stages,
        config.n_micro,
        "half",
        f_mid, b_mid, w_mid, c
    )
    local_order = pp_graph.create_schedule(config)
    print(f"OLD: " + "=" * 50)
    old = old_run_passes(config, copy.deepcopy(local_order), post_validation=True, validate=False)
    print(f"NEW: " + "=" * 50)
    new = new_run_passes(config, copy.deepcopy(local_order), post_validation=True)
    assert len(old) == len(new)
    # stage = 0
    # for old_stage_nodes, new_stage_nodes in zip(old, new):
    #     assert len(old_stage_nodes) == len(new_stage_nodes)
    #     for o, n in zip(old_stage_nodes, new_stage_nodes):
    #         assert o.get_key() == n.get_key(), f"stage {stage} old {o.type} {o.microbatch} new {n.type} {n.microbatch}"
    #     stage += 1
