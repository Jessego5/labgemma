#!/usr/bin/env bash
#
# Run the whole Day 4-6 sequence unattended.
#
#   tmux new -s all
#   ./run_all.sh
#   Ctrl-B then D
#
# These steps CANNOT run concurrently on one GPU: each loads Gemma 3 4B
# (~8.6 GB) plus activations, and a single training job already fills a 24 GB
# card. They are also compute-bound rather than latency-bound, so time-sharing
# one GPU would finish no sooner than running them in order.
#
# Every step is skipped if its output already exists, so re-running after a
# crash, a pod restart, or a Ctrl-C picks up where it stopped.
#
# For genuine parallelism, rent a 2-GPU pod and run the two fits as:
#   CUDA_VISIBLE_DEVICES=0 python -m pipeline.train fit --arm a1 &
#   CUDA_VISIBLE_DEVICES=1 python -m pipeline.train fit --arm a2 &
#   wait
# That halves the training wall clock; nothing else here benefits.

set -u
D=${LABGEMMA_DATA:-/workspace/labgemma}
RUNS=$D/runs
LOGS=$D/logs
mkdir -p "$LOGS" "$RUNS"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

step () {          # step <name> <sentinel-path> <command...>
  local name=$1 sentinel=$2; shift 2
  if [ -e "$sentinel" ]; then
    printf '  SKIP  %-22s (have %s)\n' "$name" "$(basename "$sentinel")"
    return 0
  fi
  printf '\n===  %-22s %s\n' "$name" "$(date '+%H:%M:%S')"
  "$@" 2>&1 | tee "$LOGS/$name.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '!!  %s FAILED (exit %d) -- see %s\n' "$name" "$rc" "$LOGS/$name.log"
    return "$rc"
  fi
  printf '===  %-22s done %s\n' "$name" "$(date '+%H:%M:%S')"
}

T0=$(date +%s)

# 1. Baseline evals -- no training, and the masked one fills the biggest blank
#    in the finding, so it runs before anything expensive.
step zeroshot        "$D/preds_zeroshot.json" \
  python -m pipeline.train score --out "$D/preds_zeroshot.json" || exit 1
step zeroshot_masked "$D/preds_zeroshot_masked.json" \
  python -m pipeline.train score --masked --out "$D/preds_zeroshot_masked.json" || exit 1

# 2. The two arms. Sentinel is the adapter config, which is only written on a
#    completed fit, so a killed run re-fits rather than being treated as done.
for arm in a1 a2; do
  step "fit_$arm" "$RUNS/$arm/adapter_config.json" \
    python -m pipeline.train fit --arm "$arm" || exit 1
done

# 3. Score each arm on normal and text-masked images. The 2x2 against
#    zero-shot is what shows whether hard-negative training changed the
#    MECHANISM, which matters because overall AUROC has little headroom.
for arm in a1 a2; do
  step "score_$arm" "$D/preds_$arm.json" \
    python -m pipeline.train score --adapter "$RUNS/$arm" \
      --out "$D/preds_$arm.json" || exit 1
  step "score_${arm}_masked" "$D/preds_${arm}_masked.json" \
    python -m pipeline.train score --adapter "$RUNS/$arm" --masked \
      --out "$D/preds_${arm}_masked.json" || exit 1
done

# 4. One table.
printf '\n===  report\n'
python -m pipeline.report "$D"/preds_*.json --out "$D/results.json" \
  2>&1 | tee "$LOGS/report.log"

printf '\nALL DONE in %d min\n' $(( ($(date +%s) - T0) / 60 ))
printf 'results  %s\n' "$D/results.json"
printf 'logs     %s\n' "$LOGS"
