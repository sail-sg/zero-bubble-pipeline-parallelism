# Light-weight Library of Zero Bubble Pipeline Parallelism

## How to use
1. Install by `pip install zbpp_light`
2. Insert the following code snippet to your training script at the very beginning:
```python
import zbpp_light
zbpp_light.patch_megatron()

# Your original training script starts here
import megatron, etc
```

## Supported Frameworks
- [x] Megatron-LM
- [ ] Megatron-DeepSpeed

## Current Limitations
- Only supports ZB-H1 schedule, which reduces 2/3 1F1B bubble with same memory and communication cost.
