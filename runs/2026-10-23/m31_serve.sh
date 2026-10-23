#!/bin/bash
# m31_serve.sh start|stop|status|logs  (run from the Mac): MiniMax-M3.1-Flash TP4 over the 4 Sparks, API on Bluey :8003.
# Env overrides are passed through to m31_node.sh: GMU MAXLEN MAXSEQS SPEC COMPILE_CFG HCA EXTRA_ENV NCCL_DEBUG.
set -u
here=$(cd "$(dirname "$0")" && pwd)
S4="spark-4"   # ssh aliases: spark-bluey (rank 0), spark-4, spark-asusi, spark-reddie
# rank : ssh target : mode
NODES=("spark-bluey|image" "$S4|${S4_MODE:-image}" "spark-asusi|overlay" "spark-reddie|overlay")
PASS="MAXBATCH=${MAXBATCH:-8192} GMU=${GMU:-0.85} MAXLEN=${MAXLEN:-300000} MAXSEQS=${MAXSEQS:-8} HCA=${HCA:-rocep1s0f0} NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
[ -n "${SPEC:-}" ] && PASS="$PASS SPEC='$SPEC'"
[ -n "${COMPILE_CFG:-}" ] && PASS="$PASS COMPILE_CFG='$COMPILE_CFG'"
[ -n "${EXTRA_ENV:-}" ] && PASS="$PASS EXTRA_ENV='$EXTRA_ENV'"
x() { ssh -o ConnectTimeout=15 -o BatchMode=yes $1 "$2" 2>&1 | grep -v -i "tailscale ssh"; }
case ${1:?start|stop|status|logs} in
  guard)   # (re)start the per-node memory guard (kills m31-tp4 under GUARD_MB MemAvailable)
    for r in 0 1 2 3; do h=${NODES[$r]%|*}
      ssh -o BatchMode=yes $h "mkdir -p /var/tmp/m31flash/logs; cat > /var/tmp/m31flash/m31_guard.sh; chmod +x /var/tmp/m31flash/m31_guard.sh" < "$here/m31_guard.sh"
      x "$h" "nohup setsid /var/tmp/m31flash/m31_guard.sh ${GUARD_MB:-3072} >/dev/null 2>&1 < /dev/null &"
      x "$h" 'sleep 2; echo "$(hostname) guard: $(ps -eo pid,args | grep "[m]31_guard" | grep -v -E "bash -c|be-child" | awk "{print \$1}" | head -1)"'
    done ;;
  guardstop)
    for r in 0 1 2 3; do x "${NODES[$r]%|*}" 'touch /var/tmp/m31flash/guard.stop; sleep 2; echo "$(hostname) guard left: $(ps -eo pid,args | grep "[m]31_guard.sh" | awk "{print \$1}")"; tail -1 /var/tmp/m31flash/logs/guard.log'; done ;;
  start)
    for r in 0 1 2 3; do x "${NODES[$r]%|*}" "docker rm -f m31-tp4 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"; done
    for r in 0 1 2 3; do
      h=${NODES[$r]%|*}; m=${NODES[$r]#*|}
      ssh -o BatchMode=yes $h "mkdir -p /var/tmp/m31flash/logs /var/tmp/m31flash/cache; cat > /var/tmp/m31flash/m31_node.sh" < "$here/m31_node.sh"
      x "$h" "chmod +x /var/tmp/m31flash/m31_node.sh; $PASS /var/tmp/m31flash/m31_node.sh $r $m ${EXTRA_ARGS:-}"
      # page-cache reclaim stalls GB10 loads: drop caches every 20 s for 15 min while weights stream in
      x "$h" "nohup setsid bash -c 'for i in \$(seq 1 45); do sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 20; done' >/dev/null 2>&1 < /dev/null &"
    done ;;
  stop)
    for r in 0 1 2 3; do x "${NODES[$r]%|*}" 'docker rm -f m31-tp4 >/dev/null 2>&1; echo "$(hostname) m31-tp4 removed"'; done ;;
  status)
    for r in 0 1 2 3; do x "${NODES[$r]%|*}" 'echo "$(hostname) $(docker inspect -f "{{.State.Status}} {{.State.ExitCode}}" m31-tp4 2>/dev/null || echo none) avail=$(free -m | awk "/Mem/{print \$7}")MB guard=$(ps -eo pid,args | grep "[m]31_guard.sh" | awk "{print \$1}" | head -1) $(tail -1 /var/tmp/m31flash/logs/guard_mem.log 2>/dev/null)"'; done
    curl -s -m 5 http://spark-bluey:8003/v1/models | head -c 300; echo ;;
  logs)
    for r in 0 1 2 3; do echo "=== rank $r"; x "${NODES[$r]%|*}" "docker logs --tail ${2:-15} m31-tp4 2>&1 | grep -v -E 'deprecated|opaque'"; done ;;
esac
