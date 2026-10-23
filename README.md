# MiniMax-M3.1-Flash on 4x NVIDIA DGX Spark: TP4, vLLM, DSpark, 300K context

A tested recipe for serving [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (about 428B total / 23B active MoE, text + image input) across four DGX Sparks (GB10) with tensor parallel 4. It uses the official vLLM arm64 image, MiniMax's own DSpark speculative decoding, and a 300,000-token context window. The weights are read from one node over NFS.

Measured on our four Sparks on 2026-10-23:

| | |
|---|---|
| KV pool | **1,415,656 tokens** at `--gpu-memory-utilization 0.80`, which is **4.72 full 300K requests** at once ([memory section](#memory-how-high-can-you-pin-it)) |
| Single user | 22 tok/s on a short chat prompt; 33 tok/s with a 32K prompt and 31 tok/s with a 97K prompt in context; TTFT 0.5 s on a short prompt |
| Six users | 84 tok/s total decode on short prompts (TTFT 1.2 s); six ~100K-token prompts sent at once all complete: first-token p50 221 s, last 379 s (prefill is shared, about 1,480 tok/s) |
| Prefill | about 1,480 tok/s from 8K to 112K tokens: first token of a 97K prompt in 66 s |
| DSpark | k = 3, 2.1 to 3.2 tokens per step (acceptance 0.36 on free-form prose up to 0.74 on code) |
| 69-scenario tool eval | running now on this exact config; the score lands here when it finishes |
| Checks | chat, tool call + tool-result round trip, image input, needle in a 25.8K-token prompt: all pass |

Full speed table: [C1 to C6 on four prompt sizes](#speed-c1-to-c6). Raw JSON: [`results/`](results/).

> This repo has no weights. The model is under the [MiniMax community license](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE); read it before you deploy.

## Hardware and software

| | |
|---|---|
| Nodes | 4x NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory, 121.7 GiB visible to Linux), Ubuntu 24.04.4, kernels 6.17.0-1014/-1021-nvidia, drivers 580.142 / 580.159.03 (CUDA 13.0) |
| Fabric | one QSFP cable per Spark into a MikroTik CRS812 (200G per port), RoCE v2, GID index 3, subnet 192.168.192.0/24 |
| Engine | `vllm/vllm-openai:minimax-m31-flash-arm64-cu134` (digest `sha256:5d76347628973fb80519a919a2f2740b6645b62cba3fe2bd87b25f111bd0e948`): vLLM `0.1.dev23341+g8744dd2fb` (built with CUDA 13.4), torch 2.14.0+cu130, FlashInfer 0.7.2, Triton 3.8.0, NCCL 2.30.7, transformers 5.19.0 |
| Model | MiniMaxAI/MiniMax-M3.1-Flash at revision `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`, 72 files, 252,446,157,295 bytes, DSpark draft in `dspark/` |
| Docker | 29.2.1 with the NVIDIA container toolkit |

### What vLLM runs on GB10

The model ships NVFP4 routed experts, MXFP8 attention and dense layers, and MiniMax Sparse Attention (MSA). The fast MSA kernels and the NVFP4 KV cache in this build are for SM100 (B200/GB200/B300). On GB10 (sm_121) vLLM picks these paths by itself, and the startup log shows them:

| part | kernel on GB10 |
|---|---|
| routed experts (NVFP4) | Marlin, weight-only FP4 (W4A16) |
| attention / dense linears (MXFP8) | FlashInfer CUTLASS MXFP8 |
| sparse attention and its indexer | Triton |
| KV cache, indexer cache | fp8 (`--kv-cache-dtype fp8`, `indexer_kv_dtype: fp8`); `nvfp4` needs SM100 |
| DSpark draft attention | Triton (`TRITON_ATTN`) |
| vision tower | FlashAttention |

So the GB10 numbers here are a baseline. Native FP4 MoE and MSA kernels for sm_121 would be the next speedups.

## Layout

```
node A (rank 0, API head)  /var/tmp/models/MiniMax-M3.1-Flash   <- the only copy of the weights, exported over NFS
node B, C, D (ranks 1-3)   docker NFS volume -> /models/MiniMax-M3.1-Flash (read-only, no host mount needed)
every node                 container m31-tp4, host network, /dev/infiniband, one GB10 each
API                        http://<node A>:8003/v1, model id minimax-m3.1-flash
```

Loading reads about 60 GB per rank. Over NFS on the 200G fabric that took 193 to 205 seconds.

## Quick start

1. **Download once, on the node that will export it** (about 15 minutes at ~110 MB/s on our line; the Xet transport stalled on our Sparks, so it is off):
   ```bash
   HF_HUB_DISABLE_XET=1 hf download MiniMaxAI/MiniMax-M3.1-Flash --local-dir /var/tmp/models/MiniMax-M3.1-Flash
   python3 scripts/verify_download.py MiniMaxAI/MiniMax-M3.1-Flash /var/tmp/models/MiniMax-M3.1-Flash   # sizes + sha256 of every LFS file
   ```
2. **Export it over NFS** to the fabric subnet (`/etc/exports`: `/var/tmp/models 192.168.192.0/24(rw,sync,no_root_squash,no_subtree_check,insecure)`, then `sudo exportfs -ra`). The other nodes do not mount anything; `node_run.sh` creates a docker NFS volume.
3. **Pull the image on every node**: `docker pull vllm/vllm-openai:minimax-m31-flash-arm64-cu134` (11.5 GB download, about 28 GB on disk). If a node has no room, see [overlay mode](#a-node-without-disk-for-the-image).
4. **Configure**: `cp configs/config.env.example configs/config.env` and set `NODES` (rank 0 first), `HEAD_IP`, `MODEL_DIR`/`MODEL_NFS`, `NET_IF`, `HCA` and `GID_INDEX` for your fabric.
5. **Start**: `scripts/serve.sh start`. It starts a memory guard on every node, drops the page cache, starts the four ranks and waits for `/v1/models`. First boot about 7 minutes: weights about 3.5 minutes, then KV sizing, torch.compile and CUDA graphs.
6. **Check**: `python3 scripts/smoke.py http://<head>:8003 minimax-m3.1-flash <some.jpg> <some-text-file>` (chat, tool call, image, ~32K needle), then `python3 scripts/speed_bench.py --base http://<head>:8003 --model minimax-m3.1-flash --corpus <dir with .py files> --out results.json`.
7. **Stop**: `scripts/serve.sh stop`. `scripts/serve.sh status` shows each node's MemAvailable and the guard's lowest reading.

### The serve command

What `scripts/node_run.sh` runs on each node (rank 0 shown; ranks 1-3 add `--headless` instead of `--host/--port` and use their own `--node-rank`):

```bash
vllm serve /models/MiniMax-M3.1-Flash --served-model-name minimax-m3.1-flash --trust-remote-code \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --nnodes 4 --node-rank 0 --master-addr 192.168.192.1 --master-port 29511 \
  --max-model-len 300000 --gpu-memory-utilization 0.80 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --block-size 128 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --disable-hybrid-kv-cache-manager \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
  --host 0.0.0.0 --port 8003
```

Environment inside each container (the parts that matter): `LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat:...` (see gotcha 1), `VLLM_HOST_IP=<this node's fabric IP>`, `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` = the fabric netdev, `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_NET_PLUGIN=none`, `VLLM_FLOAT32_MATMUL_PRECISION=high` (from vLLM's GB200 recipe). Docker flags: `--network host --ipc host --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK`.

The DSpark draft loads from the checkpoint's `dspark/` folder by itself; no draft path is needed. Thinking is adaptive in the chat template; set `reasoning_effort` (`low` to `max`) per request through `chat_template_kwargs` if you want to steer it. MiniMax recommends `temperature=1.0`, `top_p=0.95`.

## Memory: how high can you pin it

GB10 memory is shared by the GPU and Linux, so `--gpu-memory-utilization` is a fraction of the whole box. Everything outside vLLM's budget (Linux, docker, the API server and engine core on rank 0, JIT compilers) has to fit in the rest. Too little left and the node swaps, then stops answering ssh. We ran a ladder, each node guarded by `scripts/guard.sh`, which kills the container when MemAvailable drops under 3 GB:

| `--gpu-memory-utilization` | KV pool (tokens) | full 300K requests | MemAvailable after boot | result |
|---|---:|---:|---|---|
| 0.85 | 1,683,660 | 5.61 | 0.1 to 1.6 GB | a worker node swapped and stopped answering ssh for about 50 minutes during FlashInfer autotune (no reboot); not usable |
| 0.82 | 1,520,861 | 5.07 | 5.2 GB on rank 0, 6.7 to 6.9 GB on workers | served the smoke tests, then the guard on rank 0 fired at 3,071 MB during the first 8K prompt |
| **0.80** | **1,415,656** | **4.72** | 7.7 GB on rank 0, 9.2 to 9.4 GB on workers | **stable**: all 84 speed-test requests (up to six 100K prompts at once) and the smoke tests ran with no guard trip; lowest MemAvailable seen 4.8 GB on rank 0, 6.6 to 7.0 GB on the workers |

Rank 0 always has about 2 GB less than the workers: it also runs the API server and the engine core. Keep the guard running whatever value you pick.

## Speed: C1 to C6

Measured with [`scripts/speed_bench.py`](scripts/speed_bench.py) on the running server at `--gpu-memory-utilization 0.80`, DSpark on (k=3). Settings: temperature 1.0, top_p 0.95 (MiniMax's recommendation), 256 output tokens per request, streaming. Every request starts with a random nonce, so the prefix cache never helps. All requests of a cell start at the same moment. The client ran on the head node. The prompts are a fixed instruction (short) or slices of Python source of the requested size followed by "explain this code" (8K / 32K / 100K; the token counts vary with the slice). One warm-up request per size ran first. 84 requests, 84 OK. Raw data: [`results/sweep_gmu080.json`](results/sweep_gmu080.json), log [`results/sweep_gmu080.log`](results/sweep_gmu080.log).

| prompt | users | prompt tokens (median) | TTFT p50 (s) | prefill tok/s | decode tok/s per user | decode tok/s, all users | total tok/s | DSpark acceptance | tokens per step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 238 | 0.52 | 460 | 22.2 | 22.3 | 41 | 0.36 | 2.09 |
| short | 2 | 237 | 0.49 | 958 | 21.4 | 40.7 | 75 | 0.43 | 2.28 |
| short | 3 | 238 | 0.59 | 1,218 | 16.9 | 48.8 | 91 | 0.38 | 2.14 |
| short | 4 | 236 | 0.68 | 1,393 | 16.9 | 65.9 | 122 | 0.42 | 2.26 |
| short | 5 | 236 | 1.33 | 885 | 13.2 | 61.9 | 117 | 0.37 | 2.12 |
| short | 6 | 238 | 1.22 | 1,169 | 16.0 | 84.2 | 159 | 0.40 | 2.19 |
| 8k | 1 | 10,315 | 10.77 | 958 | 22.8 | 22.9 | 482 | 0.58 | 2.75 |
| 8k | 2 | 7,768 | 7.93 | 1,499 | 21.6 | 34.0 | 781 | 0.73 | 3.20 |
| 8k | 3 | 7,203 | 27.73 | 677 | 15.2 | 24.1 | 505 | 0.58 | 2.74 |
| 8k | 4 | 7,444 | 13.34 | 1,488 | 13.4 | 36.4 | 942 | 0.65 | 2.95 |
| 8k | 5 | 7,623 | 15.85 | 1,491 | 11.5 | 38.8 | 1,018 | 0.65 | 2.96 |
| 8k | 6 | 7,722 | 24.39 | 1,494 | 12.1 | 36.4 | 1,069 | 0.58 | 2.75 |
| 32k | 1 | 27,452 | 18.13 | 1,514 | 33.2 | 33.3 | 1,073 | 0.73 | 3.19 |
| 32k | 2 | 31,635 | 35.06 | 1,480 | 20.4 | 20.7 | 1,223 | 0.74 | 3.22 |
| 32k | 3 | 29,828 | 43.73 | 1,495 | 9.4 | 15.6 | 1,278 | 0.73 | 3.19 |
| 32k | 4 | 34,890 | 60.30 | 1,492 | 6.1 | 13.0 | 1,306 | 0.57 | 2.72 |
| 32k | 5 | 29,877 | 60.21 | 1,494 | 5.1 | 14.3 | 1,344 | 0.71 | 3.13 |
| 32k | 6 | 27,794 | 68.40 | 1,496 | 4.8 | 14.9 | 1,355 | 0.70 | 3.11 |
| 100k | 1 | 97,259 | 65.96 | 1,474 | 31.0 | 31.1 | 1,314 | 0.69 | 3.06 |
| 100k | 2 | 112,398 | 117.91 | 1,476 | 12.9 | 6.4 | 1,373 | 0.48 | 2.44 |
| 100k | 3 | 98,544 | 137.36 | 1,444 | 3.4 | 5.3 | 1,369 | 0.64 | 2.92 |
| 100k | 4 | 98,416 | 183.07 | 1,476 | 2.9 | 4.9 | 1,420 | 0.68 | 3.04 |
| 100k | 5 | 103,030 | 216.43 | 1,476 | 1.8 | 4.5 | 1,429 | 0.64 | 2.93 |
| 100k | 6 | 90,084 | 221.46 | 1,477 | 1.6 | 4.7 | 1,434 | 0.68 | 3.04 |

How to read it:
- **prefill tok/s** = all prompt tokens of the cell / time until the last request's first token. It sits at about 1,480 tok/s from 8K to 112K tokens and does not grow with users: prompts are prefilled in 8,192-token chunks, one after another.
- **decode tok/s per user** = output tokens after the first / time from first to last token (median over users). **decode tok/s, all users** = all output tokens / time from the first first-token to the last token of the cell. With long prompts and several users that window includes the other users' prefill (chunked prefill runs between decode steps), so it shows what users actually wait for. It is not steady-state decode. Steady-state decode with several users is in the short rows: 84 tok/s at 6 users.
- **total tok/s** = (prompt + output tokens) / wall time of the cell.
- **DSpark acceptance** = accepted / drafted tokens over the cell, from vLLM's `/metrics`. **tokens per step** = 1 + accepted per draft (k = 3, so at most 4). Free-form prose at temperature 1.0 accepts the fewest drafts (0.36 to 0.43); explaining code accepts 0.58 to 0.74. That is why one user decodes 22 tok/s on the short prompt but 31 to 33 tok/s with a 32K or 100K prompt in context.
- Lowest MemAvailable seen during the whole sweep (10-second samples): 4.8 GB on rank 0, 6.6 to 7.0 GB on the workers. The 3 GB guard never fired.

## Gotchas

1. **The CUDA 13.4 image on a CUDA 13.0 driver.** The image's vLLM is built with CUDA 13.4; the GB10 driver is 580 (CUDA 13.0). In image mode the NVIDIA container toolkit loads the image's forward-compat `libcuda` (615.71) and everything works. If a container ends up on the host `libcuda` instead (our overlay mode did), the Marlin NVFP4 repack fails at load with `cudaErrorUnsupportedPtxVersion` ("the provided PTX was compiled with an unsupported toolchain"). Fix: put `/usr/local/cuda-13.4/compat` first in `LD_LIBRARY_PATH` (`node_run.sh` does). Check with `cuDriverGetVersion`: it must say 13040.
2. **FlashInfer autotune.** With autotune on, the MXFP8 GEMM tuning ran over 25 minutes on GB10 (more than 218 shape tunings, all ranks in lockstep) and needs extra memory while it runs. `--kernel-config '{"enable_flashinfer_autotune":false}'` skips it; boots then take about 7 minutes.
3. **Memory headroom**, above. Rank 0 needs the most. The first long prompt also JIT-compiles a few kernels (`_prepare_dflash_inputs_kernel`, CuTeDSL `MXFP8QuantizeSwizzledKernel`); the compile cache lands in the mounted `/root/.cache`, so later boots skip it.
4. **`--block-size 128` is mandatory** for the sparse index cache, and the hybrid KV cache manager has to be off (`--disable-hybrid-kv-cache-manager`), as in vLLM's recipe.
5. **Drop the page cache before loading.** A GB10 with a full page cache stalls while 60 GB of weights stream in; `serve.sh start` drops it (needs passwordless `sudo` for `tee /proc/sys/vm/drop_caches`; skip otherwise).
6. **No Ray needed.** vLLM's mp backend spans nodes with `--nnodes/--node-rank/--master-addr` and `--headless` on the workers.
7. **Video input is off** (`--limit-mm-per-prompt '{"image":4,"video":0}'`), as in vLLM's H100 recipe, to keep the encoder's startup profiling small. Images work. We did not test video.

### A node without disk for the image

Two of our nodes had too little free disk for the 28 GB image. "Overlay" mode (`MODES` in `config.env`) runs the same bits on them: `scripts/export_runtime.sh` copies the image's `/usr/local` (python 3.12 site-packages with vLLM, CUDA 13.4 toolkit and compat libs) into the NFS export once. The node then runs the NGC `nvcr.io/nvidia/pytorch:26.07-py3` container it already has, with that `/usr/local` mounted over its own (both are Ubuntu 24.04 with python 3.12.3). It works. It is slower to page in under memory pressure, because code pages come over NFS, so prefer image mode when you have the disk.

## What failed on the way (2026-10-23)

| boot | change | what happened |
|---|---|---|
| 1 | GMU 0.85, two overlay nodes | overlay ranks loaded the host `libcuda` (CUDA 13.0): `cudaErrorUnsupportedPtxVersion` in the Marlin repack (gotcha 1) |
| 2 | + compat `libcuda` | loaded, KV pool 1,683,660, then FlashInfer autotune for 25+ minutes with almost no free memory; one node stopped answering ssh until its rank was killed |
| 3 | GMU 0.82, autotune off, memory guards | up in 7 minutes, smoke tests pass, guard fired on rank 0 during the first 8K prompt |
| 4 | GMU 0.80 | stable through the whole speed test: this is the configuration in this repo |

## Repository layout

| path | what |
|---|---|
| `scripts/serve.sh` | `start / stop / status / logs` for the four nodes, driven by `configs/config.env` |
| `scripts/node_run.sh` | one rank on one node (image or overlay mode) |
| `scripts/guard.sh` | per-node memory guard: kills `m31-tp4` under 3 GB MemAvailable |
| `scripts/export_runtime.sh` | copies the image's `/usr/local` into an NFS export, for overlay nodes |
| `scripts/verify_download.py` | checks a download against the Hugging Face API (sizes and sha256) |
| `scripts/smoke.py` | chat, tool call round trip, image input, long-prompt needle |
| `scripts/speed_bench.py` | concurrency sweep: TTFT, prefill, per-stream and aggregate decode, DSpark acceptance |
| `configs/config.env.example` | the site settings |
| `results/` | raw JSON and logs of the runs above |
| `runs/2026-10-23/` | the exact launch scripts behind these numbers, with our hostnames and paths (`m31_serve.sh`, `m31_node.sh`, `m31_guard.sh`); `scripts/` is the same thing made configurable |

## Credits

- [MiniMax](https://huggingface.co/MiniMaxAI) for MiniMax-M3.1-Flash, its DSpark draft and the quantization-aware-trained checkpoint.
- The [vLLM](https://github.com/vllm-project/vllm) project for MiniMax-M3 / M3.1 support, the arm64 image and the [MiniMax-M3.1-Flash recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash) this one starts from; [FlashInfer](https://github.com/flashinfer-ai/flashinfer), Triton and NCCL underneath it.
- NVIDIA for the DGX Spark platform and the NGC PyTorch container used in overlay mode.
- The 69-scenario tool eval follows the Pass/Partial/Fail method of wolttam's `tool-eval-bench`.

## License

The scripts and docs in this repository are MIT ([LICENSE](LICENSE)). The model weights are not included and are under MiniMax's own license.
