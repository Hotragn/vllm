#!/usr/bin/env bash
# =============================================================================
# GPU verification for vllm-project/vllm PR #54283
#   "[Bugfix][Multimodal] Frame the multi-modal hash digest input"
#
# DarkLight1337 asked for a GPU-based `vllm bench mm-processor` run to show the
# framing change does not regress multi-modal preprocessing. This script produces
# exactly that, as a paired A/B, on ONE GPU, in a single unattended pass.
#
# Why this is cheap: the PR changes exactly ONE runtime file,
# vllm/multimodal/hasher.py (the other two files are tests). With an editable
# install, swapping that file between runs changes behaviour with NO rebuild, so
# both sides share one install, one model download and one GPU.
#
# Usage (any Linux box with one NVIDIA GPU; L4 / A10 / T4 / RTX all fine):
#     bash gpu_verify_54283.sh
#
# On Colab / Kaggle, in a cell:
#     !bash gpu_verify_54283.sh 2>&1 | tail -80
#
# Tunables (env vars, all optional):
#     MODEL=Qwen/Qwen2-VL-2B-Instruct   small VL model; fits in ~8 GB
#     REPEATS=3                         measured repeats per side (+1 discarded)
#     NUM_PROMPTS=32                    requests per bench invocation
#     DTYPE=auto                        set to float16 on a T4 (no bf16 on sm75)
#
# Output: $OUT/SUMMARY.txt  <- paste this into the PR comment
#         $OUT/*.json       <- raw per-run stats from `--output-json`
# =============================================================================
set -euo pipefail

FORK=${FORK:-https://github.com/Hotragn/vllm.git}
BRANCH_HEAD=${BRANCH_HEAD:-4774d044853553463abbe6ad0d076242c9d4bee1}
# Merge base of the PR branch with upstream/main == the "before" revision.
BASE=${BASE:-435c96f9dbdd29258cb8e0f433c5b54a00cf6b16}

SRC=${SRC:-$HOME/vllm-54283}
OUT=${OUT:-$HOME/bench-54283}
MODEL=${MODEL:-Qwen/Qwen2-VL-2B-Instruct}
REPEATS=${REPEATS:-3}
NUM_PROMPTS=${NUM_PROMPTS:-32}
NUM_WARMUPS=${NUM_WARMUPS:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
DTYPE=${DTYPE:-auto}
GPU_UTIL=${GPU_UTIL:-0.85}

TARGET=vllm/multimodal/hasher.py

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

mkdir -p "$OUT"

# ---------------------------------------------------------------- 0. the GPU --
say "0/5  GPU"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: this script needs a GPU."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"

# ------------------------------------------------------------- 1. the source --
say "1/5  source at the PR head"
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none "$FORK" "$SRC"
fi
cd "$SRC"
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/vllm-project/vllm.git
git fetch origin "$BRANCH_HEAD" --depth=1 2>/dev/null || git fetch origin
git fetch upstream "$BASE" --depth=1 2>/dev/null || git fetch upstream
git -c advice.detachedHead=false checkout --force "$BRANCH_HEAD"
echo "head:  $(git rev-parse HEAD)"
echo "base:  $BASE"

# Guard: if the PR ever grows a second runtime file, the file-swap A/B below is
# no longer a faithful A/B and must not be trusted.
runtime_changed=$(git diff --name-only "$BASE" HEAD -- 'vllm/**' | tr -d '\r')
[ "$runtime_changed" = "$TARGET" ] || die \
  "expected only $TARGET to change under vllm/, got: $runtime_changed"

# ------------------------------------------------------------ 2. the install --
say "2/5  editable install (precompiled wheel; no CUDA build)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
VLLM_USE_PRECOMPILED=1 uv pip install --system -e . --torch-backend=auto
uv pip install --system pandas datasets

# HARD GUARD: the whole method rests on vllm importing from $SRC. If the import
# resolves to a site-packages copy, swapping the file is a silent no-op and every
# number below would be meaningless.
resolved=$(python -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
[ "$resolved" = "$(cd "$SRC" && pwd)" ] || die \
  "vllm imports from '$resolved', not the source tree '$SRC' -- the install is not editable."
echo "vllm imports from: $resolved  (editable, file swap is live)"

# ------------------------------------------------------- 3. the two revisions --
say "3/5  extract both revisions of $TARGET"
git show "$BASE:$TARGET"  > "$OUT/hasher.main.py"
git show "HEAD:$TARGET"   > "$OUT/hasher.branch.py"
cmp -s "$OUT/hasher.main.py" "$OUT/hasher.branch.py" && die \
  "the two revisions of $TARGET are identical -- wrong BASE?"
wc -l "$OUT/hasher.main.py" "$OUT/hasher.branch.py"

restore() { cp -f "$OUT/hasher.branch.py" "$SRC/$TARGET"; }
trap restore EXIT

run_side() {  # run_side <main|branch> <repeat>
  local side=$1 rep=$2 json="$OUT/${1}.r${2}.json"
  cp -f "$OUT/hasher.$side.py" "$SRC/$TARGET"
  find "$SRC/vllm" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  printf '   [r%s] %-6s ' "$rep" "$side"
  # Workload mirrors the Quick Start in `vllm bench mm-processor --help`, with a
  # fixed --seed so both sides hash byte-identical media. Two images per request
  # is deliberate: the PR's tagging/length framing applies to sequence and
  # mapping containers, which a single item would barely exercise.
  if vllm bench mm-processor \
        --model "$MODEL" --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --dataset-name random-mm \
        --num-prompts "$NUM_PROMPTS" --num-warmups "$NUM_WARMUPS" \
        --random-input-len 300 --random-output-len 40 \
        --random-mm-base-items-per-request 2 \
        --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
        --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}' \
        --seed 0 --disable-tqdm \
        --metric-percentiles 50,99 \
        --output-json "$json" > "$OUT/${side}.r${rep}.log" 2>&1; then
    echo "ok"
  else
    echo "ERROR (see $OUT/${side}.r${rep}.log)"
    tail -25 "$OUT/${side}.r${rep}.log" >&2
    die "bench run failed on side=$side repeat=$rep"
  fi
}

# --------------------------------------------------------------- 4. the runs --
# Repeat 0 is a discarded warmup: the first workload on a fresh GPU/page cache is
# systematically faster or slower than the rest, and whichever side runs first
# would otherwise win. Measured repeats ALTERNATE which side goes first, so any
# residual position bias cancels instead of being attributed to the diff.
say "4/5  benchmark: 1 discarded warmup + $REPEATS measured repeats, alternating order"
for rep in $(seq 0 "$REPEATS"); do
  if [ $((rep % 2)) -eq 0 ]; then order="main branch"; else order="branch main"; fi
  echo " repeat $rep  (order: $order)$([ "$rep" = 0 ] && echo '  [warmup, discarded]')"
  for side in $order; do run_side "$side" "$rep"; done
done

# ------------------------------------------------------------ 5. the summary --
say "5/5  summary"
REPEATS="$REPEATS" OUT="$OUT" MODEL="$MODEL" NUM_PROMPTS="$NUM_PROMPTS" \
python - <<'PY' | tee "$OUT/SUMMARY.txt"
import json, os, statistics as st

out, reps = os.environ["OUT"], int(os.environ["REPEATS"])
sides = ("main", "branch")

# repeat 0 is the discarded warmup
runs = {s: [json.load(open(f"{out}/{s}.r{r}.json")) for r in range(1, reps + 1)]
        for s in sides}

order = ["get_mm_hashes_ms", "get_cache_missing_items_ms", "apply_hf_processor_ms",
         "merge_mm_kwargs_ms", "apply_prompt_updates_ms", "preprocessor_total_ms",
         "encoder_forward_ms"]
# num_encoder_calls is a count, not a duration -- reported separately below as a
# same-work check rather than mixed into a table of milliseconds.
stages = [k for k in runs["main"][0].get("mm_processor_stats", {})
          if k != "num_encoder_calls"]
stages.sort(key=lambda s: (order.index(s) if s in order else 99, s))

def vals(side, stage, field):
    return [r["mm_processor_stats"][stage][field] for r in runs[side]]

print(f"vllm bench mm-processor -- PR #54283 A/B on one GPU")
print(f"model={os.environ['MODEL']}  num_prompts={os.environ['NUM_PROMPTS']}  "
      f"repeats={reps} (alternating order, 1 warmup discarded)")
print(open(f"{out}/gpu.txt").read().strip())
print()
print("Each cell is the MINIMUM across the 3 measured repeats of that repeat's")
print("per-request statistic, in ms. Lower is better; negative delta favours the PR.")
print()
hdr = (f"{'stage':<30}{'mean:main':>10}{'mean:branch':>13}{'d':>8}   "
       f"{'med:main':>10}{'med:branch':>13}{'d':>8}")
print(hdr); print("-" * len(hdr))
for stage in stages:
    row = []
    for field in ("mean", "median"):
        m, b = vals("main", stage, field), vals("branch", stage, field)
        # min-of-repeats is the least noisy estimator of true cost; the median of
        # per-run medians is reported beside it so a single fast run can't carry
        # the conclusion.
        am, ab = min(m), min(b)
        d = (ab / am - 1) * 100 if am else 0.0
        row += [am, ab, d]
    print(f"{stage:<30}{row[0]:>10.2f}{row[1]:>13.2f}{row[2]:>+7.1f}%   "
          f"{row[3]:>10.2f}{row[4]:>13.2f}{row[5]:>+7.1f}%")

print()
# Spread of main against itself, i.e. what this harness can actually resolve. A
# delta smaller than this is noise and must be reported as such, not as zero.
for stage in ("get_mm_hashes_ms", "preprocessor_total_ms"):
    if stage in stages:
        m = vals("main", stage, "mean")
        print(f"noise floor, {stage}: main measured against ITSELF varies by "
              f"{(max(m)/min(m)-1)*100:.1f}% across the {reps} repeats -- a delta "
              f"smaller than this is not resolvable by this harness.")

e2e = {s: [r["mean_e2el_ms"] for r in runs[s] if "mean_e2el_ms" in r] for s in sides}
if e2e["main"] and e2e["branch"]:
    print(f"end-to-end latency (mean, min over repeats): "
          f"main {min(e2e['main']):.1f} ms -> branch {min(e2e['branch']):.1f} ms "
          f"({(min(e2e['branch'])/min(e2e['main'])-1)*100:+.1f}%)")

# Same-work check: if the two sides did not issue the same number of encoder
# calls they did not run the same workload, and no timing comparison is valid.
calls = {s: {int(r["encoder_summary"].get("total_encoder_calls", -1))
             for r in runs[s] if r.get("encoder_summary")} for s in sides}
print(f"\nsame-work check: total encoder calls  main={sorted(calls['main'])}  "
      f"branch={sorted(calls['branch'])}"
      f"{'  <-- MISMATCH, timings not comparable' if calls['main'] != calls['branch'] else '  (equal)'}")
print(f"completed/failed requests: "
      f"main {runs['main'][0]['completed']}/{runs['main'][0]['failed']}, "
      f"branch {runs['branch'][0]['completed']}/{runs['branch'][0]['failed']}")

print("\nNOTE: random-mm generates fresh media per request, so this measures the "
      "processor-cache MISS regime -- the regime where get_mm_hashes is the "
      "smallest share of preprocessing. On a cache HIT the HF processor is "
      "skipped and hashing becomes almost all of preprocessing, so a second run "
      "with repeated media is the stricter test if this one looks clean.")
PY

say "done"
echo "paste this into the PR:   $OUT/SUMMARY.txt"
echo "raw json + logs:          $OUT"
