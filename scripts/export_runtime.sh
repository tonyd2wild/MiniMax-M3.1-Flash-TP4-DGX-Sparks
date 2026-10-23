#!/bin/bash
# export_runtime.sh DEST   (run on the NFS server node, which has IMAGE pulled)
# Copies IMAGE's /usr/local (python 3.12 dist-packages incl. vLLM, CUDA 13.4 toolkit + forward-compat libcuda) to DEST,
# for "overlay" nodes that have no disk room for the ~28 GB image. DEST must sit inside an NFS export the other nodes
# can read; set OVERLAY_NFS=<server>:DEST/usr-local in config.env. About 23 GB.
set -eu
DEST=${1:?destination directory inside an NFS export}
IMAGE=${IMAGE:-vllm/vllm-openai:minimax-m31-flash-arm64-cu134}
mkdir -p "$DEST"
cid=$(docker create "$IMAGE")
docker cp "$cid:/usr/local" "$DEST/usr-local"
docker rm "$cid" >/dev/null
du -sh "$DEST/usr-local"
