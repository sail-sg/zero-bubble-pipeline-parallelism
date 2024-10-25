import torch

import megatron.training  # For import problem
from megatron.legacy.model.recomputed_dropout import TorchRngStates, dropout


def comp_func(x, dropout_func):
    y = dropout_func(x, 0.6)
    l = y.sum()
    l.backward()
    return l, x.grad


def test_dropout_results():
    dp_funcs = [
        dropout,
        torch.nn.functional.dropout,
    ]
    ls = []
    grads = []
    test_count = 100
    for i in range(test_count):
        data = torch.rand(100, 200, dtype=torch.float32)
        if i > test_count // 2:
            data = data.cuda()
        states = TorchRngStates()
        after_states = []
        for dp_func in dp_funcs:
            states.restore()
            x = data.clone().requires_grad_(True)
            l, grad = comp_func(x, dp_func)
            ls.append(l)
            grads.append(grad)
            after_states.append(TorchRngStates())

        assert ls[0] == ls[1]
        assert torch.equal(grads[0], grads[1])
        after_states[0].assert_state_equal(after_states[1])
    print(f"dropout test passed")
