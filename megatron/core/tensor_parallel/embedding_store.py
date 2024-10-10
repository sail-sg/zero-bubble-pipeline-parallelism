from megatron.core.parallel_state import get_pipeline_model_parallel_rank

class EmbeddingStore:
    """
    For storing and retrieving intermediate results of the input embedding layer.
    """

    forward_cache = []
    backward_cache = []

    @classmethod
    def forward_store(cls, output_tensor, handle):
        cls.forward_cache.append((output_tensor, handle))

    @classmethod
    def forward_get(cls, remove=True):
        output_tensor, handle = cls.forward_cache[0]
        if handle is not None:
            handle.wait()
        if remove:
            cls.forward_cache.pop(0)
        else:
            cls.forward_cache[0] = (output_tensor, None)
        return output_tensor

    @classmethod
    def backward_store(cls, grad_output):
        cls.backward_cache.append(grad_output)

    @classmethod
    def backward_get(cls):
        contents = cls.backward_cache[0]
        cls.backward_cache.pop(0)
        return contents
