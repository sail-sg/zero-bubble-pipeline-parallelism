import sys
def wrap_w_funcs(original_func):
    from zbpp_light.weight_grad_store import WeightGradStore
    def wrapped_func(total_input, grad_output, weight):
        from megatron import get_args
        if get_args().zero_bubble:
            WeightGradStore.put(total_input, grad_output, weight, original_func)
        else:
            original_func(total_input, grad_output, weight)
    return wrapped_func

def patch_megatron():
    assert all([not x.startswith('megatron') for x in sys.modules.keys()]), 'Please patch zbpp before importing any megatron modules.'
    import fused_weight_gradient_mlp_cuda
    assert hasattr(fused_weight_gradient_mlp_cuda, 'wgrad_gemm_accum_fp32')
    assert hasattr(fused_weight_gradient_mlp_cuda, 'wgrad_gemm_accum_fp16')
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32 = wrap_w_funcs(fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32)
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16 = wrap_w_funcs(fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16)

    import megatron.core.pipeline_parallel
    from zbpp_light.zb_schedule import get_zero_bubble_forward_backward_func
    assert hasattr(megatron.core.pipeline_parallel.schedules, 'get_forward_backward_func')
    assert hasattr(megatron.core.pipeline_parallel, 'get_forward_backward_func')
    megatron.core.pipeline_parallel.schedules.get_forward_backward_func_origin = megatron.core.pipeline_parallel.schedules.get_forward_backward_func
    megatron.core.pipeline_parallel.get_forward_backward_func_origin = megatron.core.pipeline_parallel.get_forward_backward_func
    megatron.core.pipeline_parallel.schedules.get_forward_backward_func = get_zero_bubble_forward_backward_func
    megatron.core.pipeline_parallel.get_forward_backward_func = get_zero_bubble_forward_backward_func
    

    import megatron.arguments
    
    oldfunc = megatron.arguments._add_distributed_args
    def wrapped_add_distributed_args(parser):
        parser = oldfunc(parser)
        parser.add_argument('--no-zero-bubble', action='store_false', dest='zero_bubble',
                            help='Use zero bubble pipeline parallelism.')
        return parser
    megatron.arguments._add_distributed_args = wrapped_add_distributed_args