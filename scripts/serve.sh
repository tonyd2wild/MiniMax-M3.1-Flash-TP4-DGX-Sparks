#!/bin/bash
# serve.sh start|stop|status|logs [N]   MiniMax-M3.1-Flash, vLLM TP4 over four DGX Sparks.
# Run it from any machine that can ssh to the four nodes. Site settings: configs/config.env (copy config.env.example).
#   start   start the memory guard on every node, then rank 0 (API head) and ranks 1-3, then wait for /v1/models
#   stop    remove m31-tp4 on every node and stop the guards
#   status  container state, MemAvailable and the guard's lowest reading on every node, and the model list
#   logs N  last N log lines of every rank
set -u
here=$(cd "$(dirname "$0")" && pwd)
CFG=${M31_CONFIG:-$here/../configs/config.env}
[ -r "$CFG" ] || { echo "no $CFG: copy configs/config.env.example to configs/config.env and fill it in" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
read -ra N <<< "$NODES"; read -ra M <<< "${MODES:-image image image image}"
[ ${#N[@]} = 4 ] || { echo "NODES must name four nodes, rank 0 first" >&2; exit 1; }
HEAD_HOST=${N[0]#*@}
on() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "$2"; }
PASS=$(for v in HEAD_IP MASTER_PORT API_PORT MODEL_DIR MODEL_NFS SERVED_NAME IMAGE OVERLAY_BASE OVERLAY_NFS NET_IF HCA \
              GID_INDEX IB_ADDR_RANGE GMU MAXLEN MAXSEQS MAXBATCH SPEC NCCL_DEBUG EXTRA_ENV; do printf "%s=%q " "$v" "${!v:-}"; done)

case ${1:?start|stop|status|logs} in
  start)
    for r in 0 1 2 3; do
      ssh -o BatchMode=yes "${N[$r]}" "mkdir -p /var/tmp/m31flash/logs && cat > /var/tmp/m31flash/guard.sh && chmod +x /var/tmp/m31flash/guard.sh" < "$here/guard.sh"
      on "${N[$r]}" "nohup setsid /var/tmp/m31flash/guard.sh ${GUARD_MB:-3072} >/dev/null 2>&1 < /dev/null &"
      # free the page cache first: a GB10 under page-cache pressure stalls while 60 GB of weights stream in
      on "${N[$r]}" "docker rm -f m31-tp4 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true"
    done
    for r in 0 1 2 3; do
      on "${N[$r]}" "env $PASS bash -s -- $r ${M[$r]} ${EXTRA_ARGS:-}" < "$here/node_run.sh" || { echo "rank $r failed to start" >&2; exit 1; }
    done
    echo "loading: about 3.5 min for weights over NFS, then KV sizing and CUDA graphs"
    for _ in $(seq 1 120); do
      sleep 10
      if on "${N[0]}" "curl -sf -m 5 http://127.0.0.1:$API_PORT/v1/models" | grep -q "$SERVED_NAME"; then
        on "${N[0]}" "docker logs m31-tp4 2>&1 | grep -E 'GPU KV cache size' | tail -1"
        echo "up: http://$HEAD_HOST:$API_PORT/v1  model $SERVED_NAME"; exit 0
      fi
      on "${N[0]}" "docker ps -q -f name=^m31-tp4$" | grep -q . || { echo "rank 0 exited: serve.sh logs 40" >&2; exit 1; }
    done
    echo "not up after 20 min: serve.sh logs 40" >&2; exit 1 ;;
  stop)
    for r in 0 1 2 3; do on "${N[$r]}" 'docker rm -f m31-tp4 >/dev/null 2>&1; touch /var/tmp/m31flash/guard.stop; echo "$(hostname) stopped"'; done ;;
  status)
    for r in 0 1 2 3; do
      on "${N[$r]}" 'echo "$(hostname) $(docker inspect -f "{{.State.Status}}" m31-tp4 2>/dev/null || echo none) MemAvailable=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)MB $(tail -1 /var/tmp/m31flash/logs/guard_mem.log 2>/dev/null)"'
    done
    on "${N[0]}" "curl -s -m 5 http://127.0.0.1:$API_PORT/v1/models"; echo ;;
  logs)
    for r in 0 1 2 3; do echo "=== rank $r (${N[$r]})"; on "${N[$r]}" "docker logs --tail ${2:-20} m31-tp4 2>&1"; done ;;
  *) echo "start|stop|status|logs" >&2; exit 2 ;;
esac
