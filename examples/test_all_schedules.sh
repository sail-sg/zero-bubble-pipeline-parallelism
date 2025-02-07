#!/bin/bash

if dpkg -l | grep -q libibverbs-dev; then
    echo "libibverbs is installed on this Debian-based system."
else
    echo "libibverbs is NOT installed on this Debian-based system."
    exit 1
fi

methods=(
    "1f1b"
    "1f1bv"
#    "1f1b-interleaved"
    "offload-grouped-interleaved"
    "zb"
    "zbv"
    "zbv-half"
    "zbv-min"
    "seq1f1b"
)

if [ ! -z "$WORLD_SIZE" ]; then
  # Selecting RDMA devices, excluding bond0 because it's slow
  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5
  # Using RoCE v2 protocol
  export NCCL_IB_GID_INDEX=3
  # Force TCP connections of GLOO backend to be on the same interface to avoid connection errors
  export GLOO_SOCKET_IFNAME=bond0
else
  export WORLD_SIZE=1
  export RANK=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=10086
fi

rm -rf checkpoints/

while true; do
    # Check GPU memory usage
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # Initialize a flag to indicate if any GPU memory is greater than 10
    gpu_in_use=0

    # Loop through each GPU's memory usage
    for memory in $gpu_memory; do
        if [ "$memory" -gt 5000 ]; then
            ((gpu_in_use++))
        fi
    done

    echo "GPU in use: ${gpu_in_use}"

    if [ "$gpu_in_use" -lt 2 ]; then
        break
    fi
    echo "At least one NVIDIA GPU is currently in use."
    sleep 5
done

export CUDA_DEVICE_MAX_CONNECTIONS=8
export EXIT_INTERVAL=20
export LOG_INTERVAL=1
export PROFILED=1
LOGS_DIR='./logs'

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

export run='test'

for method in "${methods[@]}"; do
    echo "Method: $method"

    name="${method}"
    export AIP_RUN_NAME="$name"

    export WORLD_SIZE=1
    export PIPELINE_SIZE=${GPUS_PER_NODE}
    export TP_SIZE=1
    export LAYERS=$(( $PIPELINE_SIZE * 2 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 2 * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=4096
    export FFN_HIDDEN_SIZE=16384
    export ATTENTION_HEADS=32
    export GQA=8
    export SEQ_LENGTH=2048
    export ENABLE_ZERO_BUBBLE=
    export INTERLEAVED_1F1B=
    export ZERO_BUBBLE_MEM_LIMIT=
    export ZERO_BUBBLE_TIMER_START=5
    export ZERO_BUBBLE_TIMER_END=10
    export ZERO_BUBBLE_V_SCHEDULE=
    export ZERO_BUBBLE_V_SCHEDULE_MEM_SETUP=
    export RECOMPUTE=
    export SYNC_OPTIMIZER=
    export DISTRIBUTED_OPTIMIZER=
    export OFFLOAD=
    export INTERLEAVE_GROUP=
    export OFFLOAD_TIME=
    export OFFLOAD_CHUNK_NUM=

    export EXTRA_OPTIONS='--allow-padding-num-layers'

    if [ $method == "zb" ]; then
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
    elif [ $method == "zbv" ]; then
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        export ZERO_BUBBLE_V_SCHEDULE=1
    elif [ $method == "zbv-half" ]; then
        export SYNC_OPTIMIZER=1
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        export ZERO_BUBBLE_V_SCHEDULE=1
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --zero-bubble-v-schedule-mem-setup half"
    elif [ $method == "zbv-min" ]; then
        export SYNC_OPTIMIZER=1
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        export ZERO_BUBBLE_V_SCHEDULE=1
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --zero-bubble-v-schedule-mem-setup min"
    elif [ $method == "1f1b" ]; then
        export INTERLEAVED_1F1B=
    elif [ $method == "1f1bv" ]; then
        export INTERLEAVED_1F1B=
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --enable-1f1b-v"
    elif [ $method == "1f1b-interleaved" ]; then
        export INTERLEAVED_1F1B=1
    elif [ $method == "offload-grouped-interleaved" ]; then
        export INTERLEAVED_1F1B=1
        export INTERLEAVE_GROUP=2
        export OFFLOAD_TIME='0.2'
        export OFFLOAD_CHUNK_NUM=4
        export OFFLOAD=1
    elif [ $method == "seq1f1b" ]; then
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --num-seq-splits 2"
    else
        echo "unknown method $method"
        exit 1
    fi

    mkdir -p ${LOGS_DIR}/$run
    echo "outputing to ${LOGS_DIR}/$run/$name.$RANK.log"
    completefile=${LOGS_DIR}/$run/$name.completed
    export MASTER_PORT=$(( $MASTER_PORT + 1 ))
    if [ -f $completefile ]; then
        echo skipping $name
        continue
    fi
    env > ${LOGS_DIR}/$run/$name.$RANK.env

    if [ $method == "offload-grouped-interleaved" ]; then
        bash examples/pretrain_offload.sh 2>&1 | tee ${LOGS_DIR}/$run/$name.$RANK.log &
    else
        bash examples/pretrain_zero_bubble.sh 2>&1 | tee ${LOGS_DIR}/$run/$name.$RANK.log &
    fi
    while true; do
        if [ -f $completefile ]; then
            sleep 10
            echo 'already complete, killing'
            pkill -f python
            sleep 10
            break
        fi
        if [ -z "$(jobs | grep Running)" ]; then
            echo 'complete'
            break
        fi
        sleep 1
    done
    touch $completefile
    wait
done
