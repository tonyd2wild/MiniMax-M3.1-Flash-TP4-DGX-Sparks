#!/bin/bash
# node_run.sh RANK MODE [extra vllm args...]   (runs ON one Spark; serve.sh feeds it over ssh with the config in the env)
# Starts container m31-tp4: one tensor-parallel rank of MiniMax-M3.1-Flash (TP4 = 4 Sparks x 1 GB10, vLLM mp backend).
#   RANK 0 = API head (serves the OpenAI API on API_PORT); RANK 1-3 = --headless workers.
#   MODE image   = IMAGE from the node's docker store.
#   MODE overlay = the same bits for a node without disk room for IMAGE: IMAGE's /usr/local, exported over NFS
#                  (scripts/export_runtime.sh), mounted over /usr/local of OVERLAY_BASE (Ubuntu 24.04 + python 3.12.3).
set -u
RANK=${1:?rank}; MODE=${2:?image|overlay}; shift 2
: "${HEAD_IP:?}" "${MASTER_PORT:?}" "${API_PORT:?}" "${MODEL_DIR:?}" "${SERVED_NAME:?}" "${IMAGE:?}" "${NET_IF:?}" "${HCA:?}"
: "${GMU:?}" "${MAXLEN:?}" "${MAXSEQS:?}" "${MAXBATCH:?}" "${SPEC:?}"
GID_INDEX=${GID_INDEX:-3}; IB_ADDR_RANGE=${IB_ADDR_RANGE:-}
MY_IP=$(ip -4 -o addr show "$NET_IF" | awk '{print $4}' | cut -d/ -f1 | head -1)
W=${WORK_DIR:-/var/tmp/m31flash}
mkdir -p "$W/cache" "$W/logs"
docker rm -f m31-tp4 >/dev/null 2>&1

nfs_vol() {  # nfs_vol NAME HOST:/EXPORT/PATH   (docker mounts NFS itself; nothing to mount on the host)
  docker volume inspect "$1" >/dev/null 2>&1 || docker volume create --driver local --opt type=nfs \
    --opt o=addr=${2%%:*},ro,vers=3,nolock,hard,proto=tcp,rsize=1048576,wsize=1048576,timeo=600 \
    --opt device=":${2#*:}" "$1" >/dev/null
}
if [ -d "$MODEL_DIR" ]; then
  MODEL_MNT=(-v "$MODEL_DIR:/models/MiniMax-M3.1-Flash:ro")
else
  : "${MODEL_NFS:?MODEL_DIR is not on this node and MODEL_NFS is empty}"
  nfs_vol m31-model "$MODEL_NFS"
  MODEL_MNT=(--mount type=volume,src=m31-model,dst=/models/MiniMax-M3.1-Flash,readonly)
fi

ENVS=(
  PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  # the image's CUDA 13.4 forward-compat libcuda must win over the host's 13.0 one (see README, gotcha 1)
  LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib
  CUDA_HOME=/usr/local/cuda TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas HOME=/root
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_FLOAT32_MATMUL_PRECISION=high
  VLLM_HOST_IP=$MY_IP GLOO_SOCKET_IFNAME=$NET_IF NCCL_SOCKET_IFNAME=$NET_IF TP_SOCKET_IFNAME=$NET_IF
  NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=$GID_INDEX NCCL_IB_ROCE_VERSION_NUM=2
  NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 NCCL_NET_PLUGIN=none
  NCCL_DEBUG=${NCCL_DEBUG:-WARN}
)
[ -n "$IB_ADDR_RANGE" ] && ENVS+=(NCCL_IB_ADDR_RANGE=$IB_ADDR_RANGE)
ARGS=(
  serve /models/MiniMax-M3.1-Flash --served-model-name "$SERVED_NAME" --trust-remote-code
  --tensor-parallel-size 4 --distributed-executor-backend mp
  --nnodes 4 --node-rank "$RANK" --master-addr "$HEAD_IP" --master-port "$MASTER_PORT"
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GMU" --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$MAXBATCH"
  --block-size 128 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}'
  --disable-hybrid-kv-cache-manager
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
  --kernel-config '{"enable_flashinfer_autotune":false}'
  --limit-mm-per-prompt '{"image":4,"video":0}'
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice
)
[ "$SPEC" != off ] && ARGS+=(--speculative-config "$SPEC")
if [ "$RANK" = 0 ]; then ARGS+=(--host 0.0.0.0 --port "$API_PORT"); else ARGS+=(--headless); fi
ARGS+=("$@")

COMMON=(--name m31-tp4 --network host --ipc host --gpus all --device /dev/infiniband
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --restart no
  -v "$W/cache:/root/.cache" -v "$W/logs:/logs" "${MODEL_MNT[@]}")
if [ "$MODE" = image ]; then
  docker run -d "${COMMON[@]}" --entrypoint env "$IMAGE" -i "${ENVS[@]}" vllm "${ARGS[@]}" >/dev/null
else
  : "${OVERLAY_NFS:?}" "${OVERLAY_BASE:?}"
  nfs_vol m31-rt "$OVERLAY_NFS"
  docker run -d "${COMMON[@]}" --mount type=volume,src=m31-rt,dst=/usr/local,readonly \
    --entrypoint env "$OVERLAY_BASE" -i "${ENVS[@]}" /usr/bin/python3 /usr/local/bin/vllm "${ARGS[@]}" >/dev/null
fi
echo "$(hostname) rank $RANK ($MODE, $MY_IP): $(docker inspect -f '{{.State.Status}}' m31-tp4 2>&1)"
