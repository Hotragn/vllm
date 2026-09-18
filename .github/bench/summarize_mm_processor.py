# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a paired `vllm bench mm-processor` A/B into one table.

Fork-only helper. Reads `$OUT/{main,branch}.r<N>.json` written by
`--output-json` and prints the table that goes into the PR comment.

Measurement discipline mirrors the GPU runbook for PR #54283 so a later GPU
run is directly comparable: repeat 0 is a discarded warmup, the measured
repeats alternate which side runs first, and every cell is the MINIMUM across
repeats (the least noisy estimator) with the median reported beside it.
"""

import json
import os

OUT = os.environ["OUT"]
REPEATS = int(os.environ["REPEATS"])
SIDES = ("main", "branch")


def _load(side: str, repeat: int) -> dict:
    with open(f"{OUT}/{side}.r{repeat}.json") as f:
        return json.load(f)


runs = {s: [_load(s, r) for r in range(1, REPEATS + 1)] for s in SIDES}

ORDER = [
    "get_mm_hashes_ms",
    "get_cache_missing_items_ms",
    "apply_hf_processor_ms",
    "merge_mm_kwargs_ms",
    "apply_prompt_updates_ms",
    "preprocessor_total_ms",
    "encoder_forward_ms",
]

# num_encoder_calls is a count, not a duration -- reported below as a same-work
# check rather than mixed into a table of milliseconds.
stages = [
    k for k in runs["main"][0].get("mm_processor_stats", {}) if k != "num_encoder_calls"
]
stages.sort(key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))


def vals(side, stage, field):
    return [r["mm_processor_stats"][stage][field] for r in runs[side]]


print("vllm bench mm-processor -- PR #54283 A/B")
print(
    f"model={os.environ.get('MODEL')}  num_prompts={os.environ.get('NUM_PROMPTS')}  "
    f"repeats={REPEATS} (alternating order, 1 warmup discarded)"
)
print(f"device=CPU  {os.environ.get('CPU_DESC', '')}")
print()
print("Each cell is the MINIMUM across the measured repeats of that repeat's")
print("per-request statistic, in ms. Lower is better; negative delta favours the PR.")
print()
hdr = (
    f"{'stage':<30}{'mean:main':>10}{'mean:branch':>13}{'d':>8}   "
    f"{'med:main':>10}{'med:branch':>13}{'d':>8}"
)
print(hdr)
print("-" * len(hdr))
for stage in stages:
    row = []
    for field in ("mean", "median"):
        m, b = vals("main", stage, field), vals("branch", stage, field)
        am, ab = min(m), min(b)
        d = (ab / am - 1) * 100 if am else 0.0
        row += [am, ab, d]
    print(
        f"{stage:<30}{row[0]:>10.2f}{row[1]:>13.2f}{row[2]:>+7.1f}%   "
        f"{row[3]:>10.2f}{row[4]:>13.2f}{row[5]:>+7.1f}%"
    )

print()
# Spread of main against itself, i.e. what this harness can actually resolve.
# A delta smaller than this is noise and must be reported as such, not as zero.
for stage in ("get_mm_hashes_ms", "preprocessor_total_ms"):
    if stage in stages:
        m = vals("main", stage, "mean")
        print(
            f"noise floor, {stage}: main measured against ITSELF varies by "
            f"{(max(m) / min(m) - 1) * 100:.1f}% across the {REPEATS} repeats -- "
            f"a delta smaller than this is not resolvable by this harness."
        )

print()
# A percentage delta is misleading when the two sides' samples interleave. State
# the raw ranges and the absolute shift, which do not depend on which repeat
# happened to be fastest.
for stage in ("get_mm_hashes_ms", "preprocessor_total_ms"):
    if stage not in stages:
        continue
    m, b = sorted(vals("main", stage, "mean")), sorted(vals("branch", stage, "mean"))
    overlap = b[0] <= m[-1] and m[0] <= b[-1]
    print(
        f"{stage}: main {m[0]:.4f}..{m[-1]:.4f}  branch {b[0]:.4f}..{b[-1]:.4f}  "
        f"(absolute shift of the minima {b[0] - m[0]:+.4f} ms/request)"
    )
    verdict = (
        "OVERLAP, so the sides are not separated by this harness"
        if overlap
        else "are DISJOINT"
    )
    print(f"  -> ranges {verdict}")

e2e = {s: [r["mean_e2el_ms"] for r in runs[s] if "mean_e2el_ms" in r] for s in SIDES}
if e2e["main"] and e2e["branch"]:
    print(
        f"end-to-end latency (mean, min over repeats): "
        f"main {min(e2e['main']):.1f} ms -> branch {min(e2e['branch']):.1f} ms "
        f"({(min(e2e['branch']) / min(e2e['main']) - 1) * 100:+.1f}%)"
    )

# Same-work check: if the two sides did not issue the same number of encoder
# calls they did not run the same workload, and no timing comparison is valid.
calls = {
    s: {
        int(r["encoder_summary"].get("total_encoder_calls", -1))
        for r in runs[s]
        if r.get("encoder_summary")
    }
    for s in SIDES
}
mismatch = (
    "  <-- MISMATCH, timings not comparable"
    if calls["main"] != calls["branch"]
    else "  (equal)"
)
print(
    f"\nsame-work check: total encoder calls  main={sorted(calls['main'])}  "
    f"branch={sorted(calls['branch'])}{mismatch}"
)
print(
    f"completed/failed requests: "
    f"main {runs['main'][0]['completed']}/{runs['main'][0]['failed']}, "
    f"branch {runs['branch'][0]['completed']}/{runs['branch'][0]['failed']}"
)

print(
    "\nNOTE: random-mm generates fresh media per request, so this measures the "
    "processor-cache MISS regime -- the regime where get_mm_hashes is the "
    "smallest share of preprocessing. On a cache HIT the HF processor is "
    "skipped and hashing becomes almost all of preprocessing."
)
