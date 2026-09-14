"""Per-stage timing of the real multi-modal preprocessing pipeline, on CPU.

`vllm bench mm-processor` builds an `LLM` and therefore needs a GPU. Its
preprocessing numbers, however, come from the renderer's own
`MultiModalTimingRegistry`, and every stage it reports except
`encoder_forward_secs` runs on CPU inside `BaseMultiModalProcessor.apply`.

This driver calls that same `apply` -- via the same `get_dummy_processor_inputs`
that memory profiling uses, so the prompt and the media are built by vLLM
itself -- and reads the same `TimingContext` stages the benchmark prints:

    get_mm_hashes / get_cache_missing_items / apply_hf_processor
    merge_mm_kwargs / apply_prompt_updates / preprocessor_total

Two regimes are measured, because they weight the hashing stage very
differently:

  miss  cache=None       -- every request runs the HF processor (first-seen
                            media, or a deployment with the processor cache off)
  hit   processor cache   -- the media is already cached, so the HF processor
                            does almost nothing and hashing is the dominant
                            cost. This is the regime where a hashing change
                            matters most, so it is reported explicitly.

The headline number is `get_mm_hashes` as a FRACTION of `preprocessor_total`,
measured inside a single process on a single revision. A ratio taken that way
is immune to the between-revision runner noise that makes absolute A/B timings
on a shared runner unreliable.
"""

import gc
import json
import statistics
import sys
import traceback

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing.context import TimingContext

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MAX_LEN = 8192
ITERS = 12
WARMUP = 3

# mm_counts passed to get_dummy_processor_inputs. One large item is the common
# single-image request; eight items is the shape where per-item hashing is
# heaviest relative to the HF processor's per-batch work.
CASES = [
    ("image x1", {"image": 1}),
    ("image x8", {"image": 8}),
    ("video x1", {"video": 1}),
]


def build(mm_counts, *, cached):
    limits = {m: max(n, 1) for m, n in mm_counts.items()}
    model_config = ModelConfig(
        MODEL,
        tokenizer=MODEL,
        max_model_len=MAX_LEN,
        limit_mm_per_prompt=limits,
        mm_processor_cache_gb=4 if cached else 0,
        seed=0,
    )
    cache = MultiModalProcessorOnlyCache(model_config) if cached else None
    processor = MULTIMODAL_REGISTRY.create_processor(model_config, cache=cache)
    mm_config = model_config.get_multimodal_config()
    inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=MAX_LEN,
        mm_counts=mm_counts,
        mm_options=mm_config.limit_per_prompt,
    )
    return processor, inputs


def measure(processor, inputs):
    """Return the per-stage seconds of each timed `apply` call."""
    per_call = []
    for i in range(ITERS + WARMUP):
        ctx = TimingContext()
        gc.collect()
        gc.disable()
        try:
            processor.apply(inputs, timing_ctx=ctx)
        finally:
            gc.enable()
        if i >= WARMUP:
            per_call.append(ctx.get_stats_dict())
    return per_call


def summarize(per_call):
    stages = sorted({k for c in per_call for k in c})
    out = {}
    for stage in stages:
        vals = [c.get(stage, 0.0) for c in per_call]
        out[stage] = {
            "median_ms": statistics.median(vals) * 1e3,
            "min_ms": min(vals) * 1e3,
        }
    total = statistics.median([c.get("preprocessor_total_secs", 0.0) for c in per_call])
    hashes = statistics.median([c.get("get_mm_hashes_secs", 0.0) for c in per_call])
    out["_hash_share_pct"] = (hashes / total * 100) if total else 0.0
    out["_n"] = len(per_call)
    return out


results = {}
for regime, cached in (("miss", False), ("hit", True)):
    for name, mm_counts in CASES:
        key = f"{name} [{regime}]"
        try:
            processor, inputs = build(mm_counts, cached=cached)
            if cached:
                # Prime the processor cache so the timed calls are hits.
                processor.apply(inputs, timing_ctx=TimingContext(enabled=False))
            summary = summarize(measure(processor, inputs))
            results[key] = summary
            total_ms = summary["preprocessor_total_secs"]["median_ms"]
            print(
                f"{key:22s} total {total_ms:9.3f} ms"
                f"  hash {summary['get_mm_hashes_secs']['median_ms']:8.3f} ms"
                f"  ({summary['_hash_share_pct']:5.2f}% of preprocessing)",
                flush=True,
            )
        except Exception:
            print(f"{key:22s} FAILED", flush=True)
            traceback.print_exc()
            results[key] = {"error": traceback.format_exc()}

with open(sys.argv[1], "w") as f:
    json.dump(results, f, indent=2)
