""" GenLIP caption-length analyzer.

Samples a WebDataset (tar shards), tokenizes captions with the GenLIP tokenizer (tiktoken ``cl100k_base`` by
default, +2 for BOS/EOS to match ``TikTokenTokenizer``), and reports per-field token-length distributions plus
recommendations for the text cap (``context_length``), padding efficiency, and NaFlex batch sizing.

It handles both layouts seen in practice:
  - ``{key}.txt``  -> a single caption field named ``txt`` (e.g. cc12m).
  - ``{key}.json`` -> every string field whose key matches ``--caption-pattern`` (default ``caption``) is
    analyzed separately (e.g. monet-shuffle has caption_original / caption_florence-2-large / ...).

Example:
    python scripts/genlip_caption_stats.py '/data/n/cc12m/cc12m-train-{0000..2175}.tar' --image-seq-len 256
    python scripts/genlip_caption_stats.py /data/m/monet-shuffle/web --num-shards 8 --image-seq-len 256
"""
import argparse
import braceexpand
import glob
import json
import os
import re
import tarfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


def resolve_shards(spec: str) -> List[str]:
    """Resolve a shard spec (brace pattern, glob, or directory) to a sorted list of tar paths."""
    if os.path.isdir(spec):
        shards = sorted(glob.glob(os.path.join(spec, "*.tar")))
    else:
        shards = []
        for part in braceexpand.braceexpand(spec):
            shards.extend(glob.glob(part))
        shards = sorted(set(shards))
    if not shards:
        raise FileNotFoundError(f"No .tar shards matched: {spec}")
    return shards


def sample_shards(shards: Sequence[str], num_shards: int) -> List[str]:
    """Pick up to ``num_shards`` shards evenly spaced across the dataset (avoids head-of-dataset bias)."""
    if num_shards >= len(shards):
        return list(shards)
    idx = np.linspace(0, len(shards) - 1, num_shards).round().astype(int)
    return [shards[i] for i in sorted(set(idx.tolist()))]


def extract_captions(
        member_name: str,
        raw: bytes,
        text_keys: Sequence[str],
        json_text_key: Optional[str],
        caption_re: re.Pattern,
        keys_override: Optional[Sequence[str]],
) -> Dict[str, str]:
    """Return ``{field_name: caption_text}`` from a text member (``--text-key``) or a ``.json`` member.

    Field names mirror the training flags so recommendations are copy-pasteable: a text member yields a field
    named by its suffix (e.g. ``txt`` -> ``--text-key txt``); a JSON field yields its key (-> ``--json-text-key``).
    """
    for key in text_keys:
        if member_name.endswith("." + key):
            return {key: raw.decode("utf-8", "replace")}
    if member_name.endswith(".json"):
        try:
            obj = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return {}
        if not isinstance(obj, dict):
            return {}
        if json_text_key is not None:
            val = obj.get(json_text_key)
            return {json_text_key: val} if isinstance(val, str) else {}
        out = {}
        for key, val in obj.items():
            if not isinstance(val, str):
                continue
            if keys_override is not None:
                if key in keys_override:
                    out[key] = val
            elif caption_re.search(key):
                out[key] = val
        return out
    return {}


def collect_lengths(
        shards: Sequence[str],
        encode_fn,
        special_tokens: int,
        text_keys: Sequence[str],
        json_text_key: Optional[str],
        caption_re: re.Pattern,
        keys_override: Optional[Sequence[str]],
        max_samples: int,
) -> Dict[str, List[int]]:
    """Tokenize captions across shards; return ``{field: [token_count, ...]}`` (incl. BOS/EOS)."""
    lengths: Dict[str, List[int]] = defaultdict(list)
    member_suffixes = tuple("." + key for key in text_keys) + (".json",)
    n_samples = 0
    for shard in shards:
        try:
            tar = tarfile.open(shard)
        except (tarfile.TarError, OSError) as exc:
            print(f"  ! skipping {shard}: {exc}")
            continue
        with tar:
            for member in tar:
                if not member.name.endswith(member_suffixes):
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                caps = extract_captions(
                    member.name, fileobj.read(), text_keys, json_text_key, caption_re, keys_override,
                )
                if not caps:
                    continue
                for field, text in caps.items():
                    lengths[field].append(len(encode_fn(text)) + special_tokens)
                n_samples += 1
                if n_samples >= max_samples:
                    return lengths
    return lengths


def percentiles(arr: np.ndarray) -> Dict[str, float]:
    ps = [50, 75, 90, 95, 99]
    out = {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    out.update(mean=float(arr.mean()), std=float(arr.std()), min=int(arr.min()), max=int(arr.max()))
    return out


def round_up(value: float, multiple: int = 8) -> int:
    return int(np.ceil(value / multiple) * multiple)


def simulate_padding(capped: np.ndarray, batch_size: int, image_seq_len: int, rounds: int = 20) -> Dict[str, float]:
    """Estimate per-batch-max text padding for the no-packing 'rows' scheme.

    Returns text-token utilization (real / padded) and combined image+text utilization, averaged over random
    batch groupings. Image tokens are treated as fully used at ``image_seq_len`` (NaFlex targets the bucket).
    """
    rng = np.random.default_rng(0)
    n = len(capped)
    if n < batch_size:
        batch_size = n
    real_text = padded_text = 0.0
    for _ in range(rounds):
        perm = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            batch = capped[perm[start:start + batch_size]]
            real_text += batch.sum()
            padded_text += batch.max() * batch_size
    text_util = real_text / max(padded_text, 1)
    mean_padded = padded_text / max(real_text, 1) * capped.mean()
    total_util = (image_seq_len + capped.mean()) / (image_seq_len + mean_padded)
    return {"text_util": text_util, "mean_padded_text": mean_padded, "total_util": total_util}


def report_field(
        field: str,
        token_counts: List[int],
        image_seq_len: int,
        batch_sizes: Sequence[int],
        text_keys: Sequence[str],
        target_batch: int,
) -> None:
    arr = np.asarray(token_counts, dtype=np.int64)
    stats = percentiles(arr)
    print(f"\n{'=' * 78}\nField: {field}   (n={len(arr)} captions, caption tokens incl. BOS/EOS)\n{'=' * 78}")
    print(f"  mean {stats['mean']:6.1f}  std {stats['std']:6.1f}  min {stats['min']:4d}  max {stats['max']:5d}")
    print(f"  p50 {stats['p50']:5.0f}  p75 {stats['p75']:5.0f}  p90 {stats['p90']:5.0f}  "
          f"p95 {stats['p95']:5.0f}  p99 {stats['p99']:5.0f}")

    print(f"\n  --naflex-max-text-tokens (cap) candidates  [image_seq_len={image_seq_len}]:")
    print(f"    {'cap':>5} {'covers':>7} {'trunc%':>7} {'avg_used':>9} {'cap_util':>9} {'total_seq':>10}")
    for pct in ("p90", "p95", "p99"):
        cap = round_up(stats[pct])
        capped = np.minimum(arr, cap)
        trunc = float((arr > cap).mean()) * 100
        cap_util = capped.mean() / cap
        print(f"    {cap:>5} {pct:>7} {trunc:>6.1f}% {capped.mean():>9.1f} {cap_util:>8.0%} "
              f"{image_seq_len + cap:>10}")

    rec_cap = round_up(stats["p95"])
    capped = np.minimum(arr, rec_cap)
    row_cost = image_seq_len + rec_cap
    print(f"\n  Recommended cap ~ p95 = {rec_cap}  ->  --naflex-max-text-tokens {rec_cap}  "
          f"(per-row cost {image_seq_len}+{rec_cap}={row_cost})")
    print(f"    --naflex-max-tokens-per-batch by target rows/GPU (random padding; bucketing improves util):")
    print(f"      {'rows':>5} {'--naflex-max-tokens-per-batch':>30} {'~text_util':>11} {'~total_util':>12}")
    for bs in batch_sizes:
        sim = simulate_padding(capped, bs, image_seq_len)
        print(f"      {bs:>5} {bs * row_cost:>30,} {sim['text_util']:>10.0%} {sim['total_util']:>11.0%}")

    # Copy-paste training flags for a representative target batch.
    if field in text_keys:
        source_flag = "" if field == "txt" else f"--text-key {field} "
    else:
        source_flag = f"--json-text-key {field} "
    budget = target_batch * row_cost
    print(f"\n  Suggested train flags (~{target_batch} rows/GPU; tune budget to VRAM):")
    print(f"    {source_flag}--naflex-seq-lens {image_seq_len} --naflex-max-text-tokens {rec_cap} \\")
    print(f"      --naflex-max-tokens-per-batch {budget} \\")
    print(f"      --naflex-length-bucketing --naflex-bucket-chunk {target_batch}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="Shard spec: brace pattern, glob, or directory containing *.tar")
    parser.add_argument("--image-seq-len", type=int, default=256, help="Image patch tokens per row (NaFlex bucket).")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of shards to sample (evenly spaced).")
    parser.add_argument("--max-samples", type=int, default=20000, help="Max samples to tokenize.")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name.")
    parser.add_argument("--special-tokens", type=int, default=2, help="Control tokens added per caption (BOS+EOS).")
    parser.add_argument("--text-key", default="txt",
                        help="Tar member suffix(es) to read as a plain-text caption (matches training "
                             "--text-key); ';'-separated alternatives allowed.")
    parser.add_argument("--json-text-key", default=None,
                        help="Analyze exactly this JSON field (matches training --json-text-key). Overrides "
                             "--caption-keys/--caption-pattern.")
    parser.add_argument("--caption-pattern", default="caption",
                        help="Regex matched against JSON keys to auto-survey caption fields (when --json-text-key unset).")
    parser.add_argument("--caption-keys", nargs="+", default=None,
                        help="Explicit JSON caption keys to survey (overrides --caption-pattern).")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512],
                        help="Candidate per-GPU row counts for the --naflex-max-tokens-per-batch table.")
    parser.add_argument("--target-batch", type=int, default=256,
                        help="Rows/GPU used in the suggested train-flags snippet (sets the example budget/chunk).")
    args = parser.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding(args.encoding)
    encode_fn = enc.encode_ordinary
    caption_re = re.compile(args.caption_pattern, re.IGNORECASE)
    text_keys = tuple(args.text_key.split(";"))

    all_shards = resolve_shards(args.data)
    shards = sample_shards(all_shards, args.num_shards)
    print(f"Dataset: {args.data}")
    print(f"  {len(all_shards)} shards total; sampling {len(shards)} (evenly spaced); "
          f"tokenizer={args.encoding}(+{args.special_tokens}); max_samples={args.max_samples}")

    lengths = collect_lengths(
        shards, encode_fn, args.special_tokens, text_keys, args.json_text_key,
        caption_re, args.caption_keys, args.max_samples,
    )
    if not lengths:
        print("\nNo captions found. Check --text-key (member suffix) or --json-text-key / --caption-keys (JSON).")
        return

    for field in sorted(lengths):
        report_field(field, lengths[field], args.image_seq_len, args.batch_sizes, text_keys, args.target_batch)

    print(f"\n{'-' * 78}")
    print("Notes:")
    print("  * 'text_util'/'total_util' are for RANDOM batching (per-batch-max padding); --naflex-length-bucketing")
    print("    raises them toward ~100%/95% (set --naflex-bucket-chunk ~ rows; --naflex-bucket-pool larger for")
    print("    wide/long-caption fields).")
    print("  * Cap is set on the CLI via --naflex-max-text-tokens (truncates captions AND feeds the row cost).")
    print("  * Batch is driven by --naflex-max-tokens-per-batch = rows * (image_seq_len + cap), NOT --batch-size;")
    print("    it bounds peak memory. Pick the largest rows/budget that fits VRAM.")


if __name__ == "__main__":
    main()
