"""Compare per-stage preprocessing timings between two revisions."""

import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
branch_tag = sys.argv[2].replace("/", "_")

STAGES = [
    "get_mm_hashes_secs",
    "get_cache_missing_items_secs",
    "apply_hf_processor_secs",
    "merge_mm_kwargs_secs",
    "apply_prompt_updates_secs",
    "preprocessor_total_secs",
]


def load(tag):
    # r0 is the warmup repeat; this runner is measurably faster on its first
    # workload and whichever revision holds that slot would win unfairly.
    runs = sorted(p for p in d.glob(f"{tag}.r*.json") if ".r0." not in p.name)
    per = [json.loads(p.read_text()) for p in runs]
    if not per:
        return {}, 0
    merged = {}
    for case in per[0]:
        if "error" in per[0][case]:
            merged[case] = {"error": True}
            continue
        merged[case] = {
            s: min(r[case][s]["min_ms"] for r in per if s in r[case])
            for s in STAGES
            if s in per[0][case]
        }
        merged[case]["_hash_share_pct"] = min(r[case]["_hash_share_pct"] for r in per)
    return merged, len(per)


a, na = load("main")
b, nb = load(branch_tag)

print("")
print(f"===== per-stage preprocessing, min over {na}/{nb} A/B repeats (ms) =====")
for case in a:
    if a[case].get("error") or b.get(case, {}).get("error"):
        print(f"\n{case}: FAILED on at least one side")
        continue
    print(f"\n{case}")
    print(f"  {'stage':30s} {'main':>10s} {'branch':>10s} {'delta':>9s}")
    for s in STAGES:
        if s not in a[case] or s not in b[case]:
            continue
        m, n = a[case][s], b[case][s]
        delta = f"{(n / m - 1) * 100:+8.1f}%" if m else "       --"
        print(f"  {s:30s} {m:10.3f} {n:10.3f} {delta}")
    print(
        f"  {'get_mm_hashes share of total':30s}"
        f" {a[case]['_hash_share_pct']:9.2f}% {b[case]['_hash_share_pct']:9.2f}%"
    )
