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
ensure_tunnel() {
  if ! tunnel_alive; then
    echo "[repro] tunnel down - restarting"
    setsid nohup ssh -o BatchMode=yes -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=8 -N -L 1294:127.0.0.1:1294 \
      "$GAUTSCHI_HOST" >> runs/xrd_tunnel.log 2>&1 < /dev/null &
    sleep 8
  fi
  tunnel_alive || { echo "[repro] FATAL: cannot reach xrootd tunnel"; exit 1; }
}

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
  ensure_tunnel
  echo "[repro] $metric $tag $(date -u +%H:%M:%S)"
  python3 -m trkperf "$metric" --file-list "$list" --dataset-tag "$tag" \
    --min-q2-tier 1 "$@" || { echo "[repro] FAIL $metric/$tag"; return 1; }
}

for pair in "clean26071 filelists/clean26071_local.txt 0" \
            "bkg26071 filelists/bkg26071_local.txt 10"; do
  set -- $pair; tag=$1; list=$2; mf=$3
  extra=(--max-file-failures "$mf")
  [[ $tag == bkg* ]] && extra+=(--cache-dir cache/bkg26071_files)
  for metric in acceptance efficiency; do
    for region in central backward forward; do
      EXTRA_REGION=$region run_metric "$metric" "$tag" "$list" \
        --region "$region" "${extra[@]}" || exit 1
    done
  done
  run_metric resolution "$tag" "$list" "${extra[@]}" || exit 1
  run_metric fake-rate "$tag" "$list" "${extra[@]}" || exit 1
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
  ensure_tunnel
  python3 -m trkperf compare --metric "${name}_26071" \
    --clean "$(json_for "$m" "$r" clean26071)" \
    --bkg "$(json_for "$m" "$r" bkg26071)" \
    || echo "[repro] compare $name failed"
done
for m in acceptance efficiency resolution fake-rate; do
  for tag in clean26071 bkg26071; do
    ensure_tunnel
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
        || echo "[repro] plot skip $m/$tag/$r"
    done
  done
done
echo "[repro] DONE $(date -u)"
