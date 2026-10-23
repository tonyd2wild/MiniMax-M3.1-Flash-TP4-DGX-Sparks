#!/bin/bash
# m31_node.sh RANK MODE [extra vllm args...]  (runs ON a Spark, fed over ssh by m31_serve.sh)
# Starts container m31-tp4: one vLLM TP rank of MiniMax-M3.1-Flash (TP4 = 4 nodes x 1 GB10, mp backend, --nnodes 4).
#   RANK 0 = API head (Bluey, holds the weights locally); RANK 1-3 = --headless workers.
#   MODE image   = the official image vllm/vllm-openai:minimax-m31-flash-arm64-cu134 from the local docker store.
#   MODE overlay = same bits, for nodes without disk room for the image: the image's /usr/local (python 3.12
#                  dist-packages + CUDA 13.4) is read over NFS from Bluey and mounted over /usr/local of the
#                  NGC pytorch:26.07 container that is already on the node (both are Ubuntu 24.04, python 3.12.3).
set -u
RANK=${1:?rank}; MODE=${2:?image|overlay}; shift 2
HEAD_IP=${HEAD_IP:-192.168.192.1}           # Bluey on QSFP rail 1
MASTER_PORT=${MASTER_PORT:-29511}
API_PORT=${API_PORT:-8003}
NFS_HOST=${NFS_HOST:-192.168.192.1}         # Bluey exports /var/tmp/models rw to 192.168.192.0/24
IMAGE=vllm/vllm-openai:minimax-m31-flash-arm64-cu134
BASE=nvcr.io/nvidia/pytorch:26.07-py3
NET_IF=enp1s0f0np0
HCA=${HCA:-rocep1s0f0}
GMU=${GMU:-0.85}
MAXLEN=${MAXLEN:-300000}
MAXSEQS=${MAXSEQS:-8}
COMPILE_CFG=${COMPILE_CFG:-'{"cudagraph_mode":"FULL_AND_PIECEWISE"}'}
SPEC=${SPEC:-'{"method":"dspark","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'}
MY_IP=$(ip -4 -o addr show "$NET_IF" | awk '{print $4}' | cut -d/ -f1 | head -1)
W=/var/tmp/m31flash
mkdir -p $W/cache $W/logs
docker rm -f m31-tp4 >/dev/null 2>&1

nfs_vol() {  # nfs_vol NAME EXPORT_SUBDIR
  docker volume inspect "$1" >/dev/null 2>&1 || docker volume create --driver local --opt type=nfs \
    --opt o=addr=$NFS_HOST,ro,vers=3,nolock,hard,proto=tcp,rsize=1048576,wsize=1048576,timeo=600 \
    --opt device=":$2" "$1" >/dev/null
}
if [ -d /var/tmp/models/MiniMax-M3.1-Flash ]; then
  MODEL_MNT="-v /var/tmp/models/MiniMax-M3.1-Flash:/models/MiniMax-M3.1-Flash:ro"
else
  nfs_vol m31-model /var/tmp/models/MiniMax-M3.1-Flash
  MODEL_MNT="--mount type=volume,src=m31-model,dst=/models/MiniMax-M3.1-Flash,readonly"
fi

ENVS=(
  PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib
  CUDA_HOME=/usr/local/cuda TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas HOME=/root
  "TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0" VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}
  VLLM_USAGE_SOURCE=production-docker-image VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_FLOAT32_MATMUL_PRECISION=high
  VLLM_HOST_IP=$MY_IP GLOO_SOCKET_IFNAME=$NET_IF NCCL_SOCKET_IFNAME=$NET_IF TP_SOCKET_IFNAME=$NET_IF
  NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2
  NCCL_IB_ADDR_RANGE=192.168.192.0/24 NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0
  NCCL_IGNORE_CPU_AFFINITY=1 NCCL_NET_PLUGIN=none NCCL_DEBUG=${NCCL_DEBUG:-WARN}
)
[ -n "${EXTRA_ENV:-}" ] && read -ra XE <<< "$EXTRA_ENV" && ENVS+=("${XE[@]}")
ARGS=(
  serve /models/MiniMax-M3.1-Flash --served-model-name minimax-m3.1-flash --trust-remote-code
  --tensor-parallel-size 4 --distributed-executor-backend mp
  --nnodes 4 --node-rank "$RANK" --master-addr "$HEAD_IP" --master-port "$MASTER_PORT"
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GMU" --max-num-seqs "$MAXSEQS"
  --max-num-batched-tokens "${MAXBATCH:-8192}" --block-size 128 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}'
  --disable-hybrid-kv-cache-manager
  --compilation-config "$COMPILE_CFG"
  --limit-mm-per-prompt '{"image":4,"video":0}'
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice
)
[ "$SPEC" != off ] && ARGS+=(--speculative-config "$SPEC")
if [ "$RANK" = 0 ]; then ARGS+=(--host 0.0.0.0 --port "$API_PORT"); else ARGS+=(--headless); fi
ARGS+=("$@")

COMMON=(--name m31-tp4 --network host --ipc host --gpus all --device /dev/infiniband
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --restart no
  -v $W/cache:/root/.cache -v $W/logs:/logs $MODEL_MNT)
if [ "$MODE" = image ]; then
  docker run -d "${COMMON[@]}" --entrypoint env "$IMAGE" -i "${ENVS[@]}" vllm "${ARGS[@]}" >/dev/null
else
  nfs_vol m31-rt /var/tmp/models/m31flash-rt/usr-local
  docker run -d "${COMMON[@]}" --mount type=volume,src=m31-rt,dst=/usr/local,readonly \
    --entrypoint env "$BASE" -i "${ENVS[@]}" /usr/bin/python3 /usr/local/bin/vllm "${ARGS[@]}" >/dev/null
fi
echo "$(hostname) rank $RANK mode $MODE ip $MY_IP: $(docker inspect -f '{{.State.Status}}' m31-tp4 2>&1)"
