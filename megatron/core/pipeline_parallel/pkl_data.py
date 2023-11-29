from auto_schedule import auto_schedule, GraphConfig, Graph
import pickle

if __name__ == '__main__':
    # 32,  96, 10419, 10207,  7715, 408, 6144, 48, 64
    p, n = 32, 96
    f, b, w, c = 10419, 10207, 7715, 408
    h, a, l = 6144, 48, 64
    s = 1024
    mem_f = 34 * h + 5 * a * s
    mem_w = - 32 * h
    mem_b = - mem_w - mem_f
    graph_config = GraphConfig(
        mem_f=mem_f,
        mem_b=mem_b,
        mem_w=mem_w,
        max_mem=2 * p * mem_f,
        cost_f=f,
        cost_b=b,
        cost_w=w,
        cost_comm=c,
        print_scaling=1
    )
    graph = Graph.build_graph(p, n, graph_config)
    cost = [graph.get_cost(i) for i in range(graph.nnodes)]
    precede = [set() for i in range(graph.nnodes)]
    vis = [False] * graph.nnodes

    def get_precede(node_id):
        if vis[node_id]:
            return precede[node_id]
        for fa in graph.parents[node_id]:
            precede[node_id].add(fa)
            precede[node_id] |= get_precede(fa)
        vis[node_id] = True
        return precede[node_id]

    for i in range(graph.nnodes):
        get_precede(i)

    stage = []
    for i in range(graph.nnodes):
        stage.append(graph.get_stage(i))

    best_time, order, complete_time = auto_schedule(
        p, n, graph_config
    )
    print(complete_time)

    with open("graph.pkl", "wb") as file:
        file.write(pickle.dumps({
            'preceed': precede,
            'cost': cost,
            'device': stage,
            'solved_local_order': order,
            'solved_time': complete_time,
        }))
