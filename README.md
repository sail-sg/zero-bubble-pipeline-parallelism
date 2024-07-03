This repository is a fork of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/). The original README can be found [here](Megatron.md).

# Zero Bubble Pipeline Parallelism & Pipeline Parallelism with Controllable Memory

Zero Bubble Pipeline Parallelism is a novel pipeline parallelism algorithm able to reduce the bubble of pipeline parallelism to almost zero while preserving synchronous semantics.

Pipeline Parallelism with Controllable Memory is a novel method to build pipeline parallelism schedules with controllable activation memory. Using this method we can significantly reduce the activation memory consumption of pipeline parallelism while maintaining the same throughput or even faster.

Check out our papers at:
* [Arxiv Version with ZBV](https://arxiv.org/abs/2401.10241)
* [ICLR Accepted version with ZB1P and ZB2P](https://openreview.net/pdf?id=tuzTN0eIO5)
* [Pipeline Parallelism with Controllable Memory](https://arxiv.org/pdf/2405.15362)

A playground for zero bubble schedulers: 
* [Zero Bubble Pipeline Parallelism Scheduler Playground](https://huggingface.co/spaces/sail/zero-bubble-pipeline-parallellism)
* [Pipeline Parallelism with Controllable Memory Scheduler Playground](https://huggingface.co/spaces/sail/pipeline-parallelism-with-controllable-memory)

**Quick settings to enable Zero Bubble:**
```
  --zero-bubble-v-schedule
  --allow-padding-num-layers
  --enable-optimizer-post-validation
```
Can also try out with
`ZERO_BUBBLE_V_SCHEDULE=1 examples/pretrain_zero_bubble.sh`

Or add another flag to control the memory consumption or V schedules:
```
  --zero-bubble-v-schedule-mem-setup half
```

**Light-weight alternative options to enable ZB H1 schedule for your own megatron fork**
* Option 1: Patch a tiny ~40 line patch to your repository as described in [zb-h1-quick-start](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/blob/zb-h1-quick-start/README.md)
* Option 2: Install our pre-built zbpp packages and enable it in your own training scripts (E.g. `pretrain_gpt.py`)
```
# installed by pip install zbpp_light
import zbpp_light
zbpp_light.patch_megatron()

import megatron
...
```

**Pushing The Parento Frontier of Throughput and Memory Forward**

Our series of schedules pushes the parento frontier of throughput and memory forward.

![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/a334e0f0-eb57-4cd5-aec8-d47b1a169597)



## Schedules
The key of achieving zero bubble is to breaking a backward pass into a $B$ pass and $W$ pass. $B$ on one stage will only depend on the $B$ on its next stage, compared to depending on both $B$ and $W$ of in 1F1B.

![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/0ab6f76c-1cf0-4962-a664-124fcb3886d6)

By controlling the lifespan of each building block, we can control and lower the activation memory.

![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/18faa6c3-59fe-42b9-b91b-0e3255fc3d9e)



### Comparision of Schedules
* 1F1B
![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/1658cba3-7fef-4c41-a227-69c6b4581f50)

* ZB1P
![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/876bd529-c454-41ab-ad85-30dfb5e1c8fa)

* ZB2P
![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/373f6a27-6a7d-4a0e-92cb-a581c2c13cd5)

* ZBV - Each device is assigned to exactly 2 chunks (virtual stages), where white text colors represent the first chunk and black text colors represent the second chunk. The sequence of dependencies among model chunks follows a ”V” shape pattern for both the forward and backward passes.

![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/1e9490a9-e593-4bda-833e-8babbaea045b)

* V-Half - half of 1F1B/ZB1P's activation memory
![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/487a3207-7e91-47d7-b040-0fe0c111f667)

* V-Min - minimum (1/3) activation memory. Notice that V-Min (and only V-Min) suffers a performance degradation when F/B/W are not balanced. In practice V-Min has similar throughput as 1F1B.
![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/38c99071-df01-488f-80e8-7e766b77ba9e)



|                                                       | 1F1B    | ZB1P     | ZB2P | ZBV  | V-Half | V-Min
| ----------------------------------------------------- | ------- | -------- | ---- | --- | --- | --- |
| Bubble Rate                                           | $(p-1)/(m+p-1)=B$ | $B/3$ | 0    | 0   | $B/2$ | $2B/3 + O(n) overhead$ |
| Activation Memory <br> (Compared to 1F1B)             | 1x       | 1x        | 2x    | 1x   | 1/2x | 1/3x |
| Pipeline Communication Volume <br> (Compared to 1F1B) | 1x       | 1x        | 1x    | 2x   | 2x   | 2x   |



<p style="font-size:14px;margin-bottom:0;height:20px;">* p: number of pipeline stages; m: number of microbatches</p>
<p style="font-size:14px;margin-bottom:0;height:20px;">* Assuming T<sub>F</sub> = T<sub>B</sub> = T<sub>W</sub></p>
<p style="font-size:14px;margin-bottom:0;height:20px;">* Communication volume of DP and TP stays the same</p>


## Zero Bubble Command Line Arguments

* `--enable-zero-bubble` Enables zero bubble schedules.
* `--zero-bubble-v-schedule` Enables V schedule recommended above. Implies `--enable-zero-bubble`.
* `--zero-bubble-v-schedule-mem-setup` Sets the memory limit for V schedules, valid options are `min`/`half`/`zb` (default).
* `--enable-optimizer-post-validation` Enables optimizer post validation explained in [Optimizer Post Validation](#Optimizer-Post-Validation)
* `--allow-padding-num-layers` Allowing the number of layers to NOT be a mutiple of number of Pipelines. This allows us to have one less layer on the first and last pipeline stage to compensate for the bubble caused by embedding layers.
* `--zero-bubble-max-pending-backward` Controls memory limit of zero bubble schedules. Setting this to 1 x number of pipelines will get a schedule like ZB1P while setting to 2x number of pipelines will get ZB2P. No effect for ZBV schedule enabled by `--zero-bubble-v-schedule`.
* `--zero-bubble-pipeline-timers-start-iter` and `--zero-bubble-pipeline-timers-end-iter` Used to control the start/end iterations when ZB scheduler profiles each F/B/W to measure $T_F$, $T_B$ and $T_W$

**Notices**
* V schedule requires the number of layers per pipeline to be an even number, so that each stage can be splited into two virtual stages evenly.
* To achieve a better throughput, we recommend setting `--num-layers` to a value to `k * pipeline-model-parallel-size - 2` where k can be any value $\ge1$. This is used to compensate for the additional embedding layer on the first/last pipeline stages which could otherwise brings bubble to all other stages.

## Optimizer Post Validation

In most practices of PP there's an all-reduce cross all pipeline stages for numerical robustness, e.g. global gradient norm for gradient clipping. INF/NAN check for mixed precision training, etc. This all-reduce breaks parallelogram and makes zero bubble impossible.
Under the observation that during a stable training both the gradient clipping and INF/NAN rarely triggers, we replace the before-hand synchronizations with a post update validation.

![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/40be4651-7240-4962-bd2a-246557752768)

We eagerly step the optimizers assuming the grad cliping, INF/NAN conditions are not triggered. In case an amendment to the gradient is required, a rollback will be issued and then we redo the optimizer step based on the fully reduced global state.

To enable this feature, add `--enable-optimizer-post-validation`. Experiments shows NOT enabling this will cause ~8% performance loss.
