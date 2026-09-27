#!/usr/bin/env bash
# Same-campaign (26.07.1) clean-vs-background reproduction from the local
# gautschi copies, streamed over an ssh-tunnelled xrootd daemon
# (root://localhost:1294//...). Sequential by design: the efficiency metric
# on the 300-file clean sample is the memory peak and must not overlap the
# others. Resumable: every metric run is skipped if its JSON already exists.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs output
export PYTHONUNBUFFERED=1

tunnel_alive() { (exec 3<>/dev/tcp/127.0.0.1/1294) 2>/dev/null; }
# NOTE (2026-09-25): pin to login00. gautschi.rcac.purdue.edu round-robins
# over 4 login nodes; the xrootd daemon binds 127.0.0.1 on ONE node, so a
# tunnel landing on any other node hangs forever. login00 is directly
# sshable; the daemon must run on the same node the tunnel lands on.
readonly GAUTSCHI_HOST=login00.gautschi.rcac.purdue.edu
readonly XROOTD_BIN=/cvmfs/oasis.opensciencegrid.org/osg/modules/xrootd/4.2.1/bin/xrootd
readonly XROOTD_LIB=/cvmfs/oasis.opensciencegrid.org/osg/modules/xrootd/4.2.1/lib64
readonly XROOTD_ROOT=/scratch/gautschi/wxie/eIC_data_small_set
# A listening port is NOT proof the data path works: a half-dead tunnel
# accepts the TCP connection but forwards nothing, and uproot then hangs
# until "Operation expired". Verify with a real read of the first file.
stack_alive() {
  tunnel_alive || return 1
  timeout 120 python3 -c "
import uproot
t = uproot.open('root://localhost:1294//${XROOTD_ROOT}/clean/' + '$(head -1 filelists/clean26071_local.txt | xargs basename)')['events']
assert t.num_entries > 0
" 2>/dev/null
}
restart_tunnel() {
  echo "[repro] (re)starting tunnel to $GAUTSCHI_HOST"
  pkill -f "ssh .* -L 1294:" 2>/dev/null
  sleep 2
  setsid nohup ssh -o BatchMode=yes -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=8 -o ExitOnForwardFailure=yes -N -L 1294:127.0.0.1:1294 \
    "$GAUTSCHI_HOST" >> runs/xrd_tunnel.log 2>&1 < /dev/null &
  sleep 8
}
restart_daemon() {
  # A stale daemon from a killed session can hold port 1294 while serving
  # nothing; clear it before starting a fresh one.
  echo "[repro] (re)starting remote xrootd daemon on $GAUTSCHI_HOST"
  timeout 60 ssh -o BatchMode=yes "$GAUTSCHI_HOST" \
    "pkill -f 'xrootd.*-p 1294'; sleep 3; mkdir -p ~/xrdlog; \
     export LD_LIBRARY_PATH=${XROOTD_LIB}:\$LD_LIBRARY_PATH; \
     setsid nohup ${XROOTD_BIN} -p 1294 -b 127.0.0.1 -l ~/xrdlog/xrootd.log \
       ${XROOTD_ROOT} < /dev/null > ~/xrdlog/xrd_start.out 2>&1 & sleep 12" \
    >> runs/xrd_tunnel.log 2>&1
}
ensure_stack() {
  # Tunnel first (cheap), daemon second (a fresh daemon needs seconds to
  # initialise), then one last tunnel check - the daemon restart can take
  # down a tunnel that was bound to the old instance's fate.
  if stack_alive; then return 0; fi
  echo "[repro] data path down - restarting tunnel"
  restart_tunnel
  if stack_alive; then return 0; fi
  echo "[repro] tunnel up but no data - restarting remote daemon"
  restart_daemon
  restart_tunnel
  stack_alive || { echo "[repro] FATAL: data path still down after tunnel+daemon restart"; return 1; }
}
ensure_tunnel() { ensure_stack; }

FAILED=()
run_metric () {  # $1 metric  $2 tag  $3 filelist  [extra args...]
  local metric=$1 tag=$2 list=$3; shift 3
  local out="output/${metric}_${tag}.json"
  case "$metric" in
    acceptance|efficiency)
      local region=${EXTRA_REGION:-central}
      [[ $region == central ]] || out="output/${metric}_${region}_${tag}.json" ;;
    *) out="output/${metric}_${tag}.json" ;;
  esac
  if [[ -f $out ]]; then echo "[repro] skip $metric/$tag (exists)"; return 0; fi
  # Stochastic in-flight drops (fsspec_xrootd segfaults, endpoint hiccups)
  # are retried; the per-file read cache makes every retry after the first
  # cheap, so many attempts converge where few would die.
  local attempt=1
  while (( attempt <= 10 )); do
    ensure_stack || { echo "[repro] FAIL $metric/$tag (stack down, attempt $attempt)"; (( attempt++ )); continue; }
    echo "[repro] $metric $tag $(date -u +%H:%M:%S) (attempt $attempt)"
    if python3 -m trkperf "$metric" --file-list "$list" --dataset-tag "$tag" \
      --min-q2-tier 1 "$@"; then
      return 0
    fi
    echo "[repro] attempt $attempt failed for $metric/$tag - retrying"
    (( attempt++ ))
  done
  echo "[repro] FAIL $metric/$tag (10 attempts exhausted)"
  FAILED+=("$metric/$tag")
  return 1
}

# Chunk sizing for the heavy clean runs: 300 files / 4 chunks of 75-80. The
# 300-file clean efficiency peaked near the box's RAM ceiling in one pass;
# quarters bound it to roughly a quarter of that. bkg runs (275 x 99-event
# files) completed in one pass and stay unchunked. Tunable, not magic.
readonly CHUNK_FILES_CLEAN=80

for pair in "clean26071 filelists/clean26071_local.txt 0" \
            "bkg26071 filelists/bkg26071_local.txt 10"; do
  set -- $pair; tag=$1; list=$2; mf=$3
  extra=(--max-file-failures "$mf")
  # Cache every tag: the 3 acceptance + 3 efficiency region runs re-read the
  # same truth/hit/reco branch-sets per file; the pickled cache turns the
  # 2nd..nth read of each (file, branch-set) into local disk I/O.
  extra+=(--cache-dir "cache/${tag}_files")
  # This driver is sequential by construction (one metric at a time), so
  # the heavy runs never overlap in memory - no wait_mem guards needed.
  for metric in acceptance efficiency; do
    for region in central backward forward; do
      # Chunk only the heavy clean side; the bkg side proved itself in one pass.
      chunk=()
      if [[ $tag == clean* && $metric == efficiency ]]; then
        chunk=(--chunk-files "$CHUNK_FILES_CLEAN")
      fi
      EXTRA_REGION=$region run_metric "$metric" "$tag" "$list" \
        --region "$region" "${extra[@]}" "${chunk[@]}" || true
    done
  done
  chunk=()
  [[ $tag == clean* ]] && chunk=(--chunk-files "$CHUNK_FILES_CLEAN")
  run_metric resolution "$tag" "$list" "${extra[@]}" "${chunk[@]}" || true
  run_metric fake-rate "$tag" "$list" "${extra[@]}" || true
done

# comparisons + per-region grouped plots
# central metrics live at output/<metric>_<tag>.json; region metrics at
# output/<metric>_<region>_<tag>.json (central keeps the legacy name).
json_for () {  # $1 metric  $2 region("" for central)  $3 tag
  if [[ -n $2 ]]; then echo "output/$1_$2_$3.json"; else echo "output/$1_$3.json"; fi
}
for spec in "acceptance none" "acceptance backward" "acceptance forward" \
            "efficiency none" "efficiency backward" "efficiency forward" \
            "resolution none" "fake-rate none"; do
  m=${spec%% *}; r=${spec#* }; [[ $r == none ]] && r=""
  name="${m}${r:+-$r}"
  clean_json="$(json_for "$m" "$r" clean26071)"
  bkg_json="$(json_for "$m" "$r" bkg26071)"
  if [[ ! -f $clean_json || ! -f $bkg_json ]]; then
    echo "[repro] compare $name skipped (missing side)"
    FAILED+=("compare-$name")
    continue
  fi
  ensure_stack || { echo "[repro] compare $name skipped (stack down)"; FAILED+=("compare-$name"); continue; }
  python3 -m trkperf compare --metric "${name}_26071" \
    --clean "$clean_json" \
    --bkg "$bkg_json" \
    || { echo "[repro] compare $name failed"; FAILED+=("compare-$name"); }
done
for m in acceptance efficiency resolution fake-rate; do
  for tag in clean26071 bkg26071; do
    ensure_stack || { echo "[repro] plots for $m/$tag skipped (stack down)"; continue; }
    for r in barrel "backward endcap" "forward endcap"; do
      case "$r" in
        barrel) base=$(json_for "$m" "" "$tag") ;;
        "backward endcap") base=$(json_for "$m" backward "$tag") ;;
        "forward endcap") base=$(json_for "$m" forward "$tag") ;;
      esac
      # resolution/fake-rate have no region-rule files: slice the central
      # JSON (same convention as the 26.02/26.07 mixed study).
      [[ $m == resolution || $m == fake-rate ]] && base=$(json_for "$m" "" "$tag")
      [[ -f $base ]] && python3 -m trkperf plot --json "$base" --grouped --eta-region "$r" \
        || { echo "[repro] plot skip $m/$tag/$r"; FAILED+=("plot-$m/$tag/$r"); }
    done
  done
done
if (( ${#FAILED[@]} )); then
  echo "[repro] DONE $(date -u) WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
echo "[repro] DONE $(date -u)"
