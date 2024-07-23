from megatron.core.pipeline_parallel.zerobubble.scheduler import zb, ScheduledNode, zbv, zbv_greedy
from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import run_schedule_passes


def check_nodes(expected, actual):
    assert len(expected) > 0
    assert len(expected) == len(actual)
    stage = 0
    for e, a in zip(expected, actual):
        assert len(e) > 0
        assert len(e) == len(a)
        for en, an in zip(e, a):
            assert isinstance(en, ScheduledNode)
            assert isinstance(an, ScheduledNode)
            assert an == en
        stage += 1


def repeat(element, num):
    return [element for _ in range(num)]


def test_zb_schedules():
    n_stages = 8
    nmb = 8

    hidden_size = 4096 * 4
    num_attention_heads = 16
    seq_length = 4096
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx
    mem_limit = f_mem_approx * n_stages

    f, b, w, c = 5, 6, 4, 1
    f_mem, b_mem, w_mem = f_mem_approx, b_mem_approx, w_mem_approx
    config = zb.GraphConfig(
            cost_f=repeat(f, n_stages),
            cost_b=repeat(b, n_stages),
            cost_w=repeat(w, n_stages),
            cost_comm=c,
            mem_f=repeat(f_mem, n_stages),
            mem_b=repeat(b_mem, n_stages),
            mem_w=repeat(w_mem, n_stages),
            max_mem=repeat(mem_limit, n_stages),
            print_scaling=1000,
        )

    expected_nodes = zb.auto_schedule(n_stages, nmb, config)

    graph, local_order = zb.create_schedule(n_stages, nmb, config)
    local_order = run_schedule_passes(graph, local_order)
    check_nodes(expected_nodes, local_order)


def test_zbv_schedules():
    n_stages = 8
    nmb = 8

    hidden_size = 4096 * 4
    num_attention_heads = 16
    seq_length = 4096
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx
    mem_limit = f_mem_approx * n_stages

    f, b, w, c = 5, 6, 4, 1
    f_mem, b_mem, w_mem = f_mem_approx, b_mem_approx, w_mem_approx
    config = zb.GraphConfig(
            cost_f=repeat(f, n_stages),
            cost_b=repeat(b, n_stages),
            cost_w=repeat(w, n_stages),
            cost_comm=c,
            mem_f=repeat(f_mem, n_stages),
            mem_b=repeat(b_mem, n_stages),
            mem_w=repeat(w_mem, n_stages),
            max_mem=repeat(mem_limit, n_stages),
            print_scaling=1000,
            max_chunks=2,
    )

    f_mid, b_mid, w_mid = f, b, w
    pp_graph = zbv.PipelineGraph(
                n_stages,
                nmb,
                f_mid, b_mid, w_mid, c,
                # V schedule does not consider memory differences between stages for now.
                f_mem=f_mem_approx, b_mem=b_mem_approx, w_mem=w_mem_approx,
                max_mem=None
                # Mem ignored for now
            )
    expected_nodes = pp_graph.get_v_schedule()

    graph, local_order = pp_graph.create_schedule(config)
    local_order = run_schedule_passes(graph, local_order)
    check_nodes(expected_nodes, local_order)


def test_zbv_greedy_schedules():
    n_stages = 8
    nmb = 8

    hidden_size = 4096 * 4
    num_attention_heads = 16
    seq_length = 4096
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx
    mem_limit = f_mem_approx * n_stages

    f, b, w, c = 5, 6, 4, 1
    f_mem, b_mem, w_mem = f_mem_approx, b_mem_approx, w_mem_approx
    config = zb.GraphConfig(
        cost_f=repeat(f, n_stages),
        cost_b=repeat(b, n_stages),
        cost_w=repeat(w, n_stages),
        cost_comm=c,
        mem_f=repeat(f_mem, n_stages),
        mem_b=repeat(b_mem, n_stages),
        mem_w=repeat(w_mem, n_stages),
        max_mem=repeat(mem_limit, n_stages),
        print_scaling=1000,
        max_chunks=2,
    )

    f_mid, b_mid, w_mid = f, b, w
    for mem_config in ['min', 'half']:
        pp_graph = zbv_greedy.PipelineGraph(
                    n_stages,
                    nmb,
                    mem_config,
                    f_mid, b_mid, w_mid, c,
                )
        expected_nodes = pp_graph.get_schedule()

        graph, local_order = pp_graph.create_schedule(config)
        local_order = run_schedule_passes(graph, local_order)
        check_nodes(expected_nodes, local_order)
