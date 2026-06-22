"""WebDataset multi-source manifest support for ``--train-data``.

A *manifest* is a small JSON (always) or YAML (when ``pyyaml`` is installed) file that lists several WebDataset
sources at once, so a training run can point ``--train-data`` at one file instead of a giant ``::``-joined shard
string plus a parallel ``--train-data-upsampling-factors`` string and a hand-computed ``--train-num-samples``.

The manifest is *resolved* into the values the existing WDS builders already read -- it never introduces a new
data pipeline. Resolution produces **flat per-shard Python lists** (not a ``::`` string): each source is
brace-expanded here and its weight replicated across its shards. This is the representation
:func:`open_clip_train.data.expand_urls` passes through verbatim (its non-string branch returns
``(list(urls), weights)``), and that both ``ResampledShards2`` (which re-runs ``expand_urls`` and asserts
``len(urls) == len(weights)``) and ``get_dataset_size`` accept -- so nothing downstream changes.

Inventory vs policy: the manifest is an *inventory* (shards + effective sample counts + intended mix weights).
Whether to sample shards with replacement is a *run* decision and stays on the CLI (``--dataset-resampled``);
the manifest does not toggle it. Sampling weights only take effect under resampling; without
``--dataset-resampled`` they are ignored (the run reads shards in source order) and a line is logged. A future
non-resampled weighting mode could approximate *integer* source ratios by duplicating a source's shards in the
list (SimpleShardList walks each shard once per epoch), but that is out of scope here.

Weighting note: under shard resampling the sample-level fraction of a source is proportional to
``sampling_weight * samples`` (a shard is visited in proportion to its weight, then drained). So to hit a target
sample fraction ``f_i`` the per-source weight must be ``f_i / samples_i`` -- this is what ``weight_mode:
"mixture_fraction"`` computes. ``samples`` should be the *effective* (post-filter) sample count, not raw archive
rows.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import braceexpand

_logger = logging.getLogger(__name__)

_MANIFEST_EXTS = (".json", ".yaml", ".yml")
_WEIGHT_MODES = ("raw", "mixture_fraction")
# Cache parsed manifests keyed by abspath so looks_like_manifest() + load_manifest() don't read twice.
_PARSE_CACHE: Dict[str, Any] = {}


def _read_manifest_file(path: str) -> Any:
    """Parse ``path`` as JSON or YAML and return the parsed object (any type).

    A *syntax error* raises a clear ValueError -- a ``.json``/``.yaml`` file the user explicitly handed to
    ``--train-data`` is meant to be a manifest, so a parse failure should be friendly, not flow downstream as a
    bogus shard path. (Valid-but-not-a-manifest content -- e.g. JSON lacking ``sources`` -- is handled by the
    callers, not here.) YAML support is soft: missing ``pyyaml`` raises a clear ImportError only for a
    ``.yaml``/``.yml`` file, so the JSON path stays zero-dependency.
    """
    key = os.path.abspath(path)
    if key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    f"Reading the YAML manifest {path!r} requires pyyaml. Install it "
                    "(`pip install pyyaml`) or convert the manifest to JSON."
                ) from e
            try:
                obj = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Manifest {path!r} is not valid YAML: {e}") from e
        else:
            try:
                obj = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Manifest {path!r} is not valid JSON: {e}") from e
    _PARSE_CACHE[key] = obj
    return obj


def looks_like_manifest(train_data: Any) -> bool:
    """Return True if ``train_data`` is a path to a manifest (mapping with a ``sources`` key).

    A normal shard spec -- a ``.tar`` glob or a ``::``-joined string -- fails the extension or the ``sources``
    check, so detection never misfires on a real WebDataset input. Note: a file with a manifest extension that
    exists but has a *syntax error* raises (via :func:`_read_manifest_file`) rather than returning False, so a
    typo'd ``mix.yaml`` fails clearly instead of being treated as a shard path.
    """
    if not isinstance(train_data, str):
        return False
    if not train_data.lower().endswith(_MANIFEST_EXTS):
        return False
    if not os.path.isfile(train_data):
        return False
    parsed = _read_manifest_file(train_data)
    return isinstance(parsed, dict) and "sources" in parsed


def load_manifest(path: str) -> Dict[str, Any]:
    """Read and validate a manifest file, returning the parsed dict.

    Raises ValueError on a parse error or any schema problem so misconfigured runs fail loudly at parse time.
    """
    manifest = _read_manifest_file(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest {path!r} is not a JSON/YAML mapping.")
    _validate_manifest(manifest, path)
    return manifest


def _validate_manifest(manifest: Dict[str, Any], path: str) -> None:
    if manifest.get("version") != 1:
        raise ValueError(f"Manifest {path!r}: 'version' must be 1, got {manifest.get('version')!r}.")
    weight_mode = manifest.get("weight_mode", "raw")
    if weight_mode not in _WEIGHT_MODES:
        raise ValueError(f"Manifest {path!r}: 'weight_mode' must be one of {_WEIGHT_MODES}, got {weight_mode!r}.")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Manifest {path!r}: 'sources' must be a non-empty list.")
    for i, src in enumerate(sources):
        where = f"Manifest {path!r}, sources[{i}]"
        if not isinstance(src, dict):
            raise ValueError(f"{where}: each source must be a mapping.")
        shards = src.get("shards")
        if not (
            isinstance(shards, str) or (isinstance(shards, list) and shards and all(isinstance(s, str) for s in shards))
        ):
            raise ValueError(f"{where}: 'shards' must be a string or non-empty list of strings.")
        if not isinstance(src.get("samples"), int) or src["samples"] <= 0:
            raise ValueError(
                f"{where}: 'samples' (effective post-filter count) is required and must be a positive int."
            )
        if weight_mode == "mixture_fraction":
            frac = src.get("mixture_fraction")
            if not isinstance(frac, (int, float)) or frac <= 0:
                raise ValueError(f"{where}: weight_mode 'mixture_fraction' requires a positive 'mixture_fraction'.")
        else:
            sw = src.get("sampling_weight", 1.0)
            if not isinstance(sw, (int, float)) or sw < 0:
                raise ValueError(f"{where}: 'sampling_weight' must be a non-negative number.")


def _expand_source_shards(shards: Any) -> List[str]:
    """Brace-expand a source's ``shards`` (str or list of str) into a flat list of concrete shard paths."""
    patterns = [shards] if isinstance(shards, str) else list(shards)
    expanded: List[str] = []
    for pat in patterns:
        expanded.extend(braceexpand.braceexpand(pat))
    return expanded


def _resolve_sources(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-source resolution: list of ``{name, shards:[concrete...], samples, weight}`` where ``weight`` is the
    pre-normalization per-source sampling weight (``mixture_fraction / samples`` for fraction mode, else
    ``sampling_weight``)."""
    weight_mode = manifest.get("weight_mode", "raw")
    rows: List[Dict[str, Any]] = []
    for src in manifest["sources"]:
        shards = _expand_source_shards(src["shards"])
        samples = int(src["samples"])
        if weight_mode == "mixture_fraction":
            # sample-fraction f ∝ weight * samples  ->  weight ∝ f / samples
            weight = float(src["mixture_fraction"]) / samples
        else:
            weight = float(src.get("sampling_weight", 1.0))
        rows.append({"name": src.get("name", "?"), "shards": shards, "samples": samples, "weight": weight})
    return rows


def _normalized_source_weights(rows: List[Dict[str, Any]]) -> List[float]:
    """Per-source weights normalized so the max is 1.0 (cosmetic; ResampledShards2 uses relative weights)."""
    max_w = max((r["weight"] for r in rows), default=0.0)
    if max_w <= 0:
        return [0.0 for _ in rows]
    return [r["weight"] / max_w for r in rows]


def _per_shard(values: List[float], rows: List[Dict[str, Any]]) -> List[float]:
    """Replicate one per-source value across each source's shards, aligned to the flattened shard list."""
    return [v for v, r in zip(values, rows) for _ in r["shards"]]


def resolve_manifest(manifest: Dict[str, Any]) -> Tuple[List[str], Optional[List[float]], int]:
    """Resolve a (validated) manifest into ``(train_data_list, sampling_weights_list_or_None, num_samples)``.

    ``train_data_list`` is every source's brace-expanded shards flattened. ``sampling_weights_list`` is the
    per-shard weight (each source's normalized weight replicated across its shards) aligned to
    ``train_data_list``, or None when raw mode is uniform (so the run behaves exactly like no upsampling
    factors). ``num_samples`` sums per-source ``samples`` -- the reliable size source for multi-directory mixes
    (``get_dataset_size`` only reads the first shard's directory).
    """
    rows = _resolve_sources(manifest)
    all_shards = [s for r in rows for s in r["shards"]]
    all_weights = _per_shard(_normalized_source_weights(rows), rows)
    total_samples = sum(r["samples"] for r in rows)
    uniform = len(set(round(w, 12) for w in all_weights)) <= 1
    weights_out = None if (manifest.get("weight_mode", "raw") == "raw" and uniform) else all_weights
    return all_shards, weights_out, total_samples


def _parse_cli_upsampling_factors(factor_str: str, source_shard_counts: List[int]) -> List[float]:
    """Expand a CLI ``--train-data-upsampling-factors`` ``::`` string into a per-shard list aligned to the
    manifest's flattened shard list. Accepts one value per source (replicated by that source's shard count) or
    one value per shard (used as-is); anything else is an error."""
    parts = [float(x) for x in factor_str.split("::")]
    num_sources = len(source_shard_counts)
    num_shards = sum(source_shard_counts)
    if len(parts) == num_sources:
        return [f for f, n in zip(parts, source_shard_counts) for _ in range(n)]
    if len(parts) == num_shards:
        return parts
    raise ValueError(
        f"--train-data-upsampling-factors has {len(parts)} values; with this manifest expected "
        f"{num_sources} (one per source) or {num_shards} (one per shard)."
    )


def apply_manifest_to_args(args, manifest: Dict[str, Any]) -> None:
    """Resolve ``manifest`` and write the results onto ``args``, applying the resampling/weight policy.

    Policy:
      * ``--dataset-type`` must be explicit (not ``auto``) -- ``auto`` would later ``.split('.')`` the now-list
        ``train_data``.
      * Resampling is a CLI-only decision (``--dataset-resampled``); the manifest does not toggle it.
      * Manifest weights apply only under resampling; without ``--dataset-resampled`` they are ignored with a
        log (the run reads shards in source order).
      * CLI ``--train-data-upsampling-factors`` overrides manifest weights, expanded here to a per-shard list so
        it matches the now-list ``train_data``; it requires ``--dataset-resampled`` and is rejected early here
        otherwise. CLI ``--train-num-samples`` overrides the manifest total.
    """
    if getattr(args, "dataset_type", None) == "auto":
        raise ValueError(
            "A --train-data manifest requires an explicit --dataset-type (webdataset or webdataset-audio), not 'auto'."
        )

    rows = _resolve_sources(manifest)
    shards = [s for r in rows for s in r["shards"]]
    source_shard_counts = [len(r["shards"]) for r in rows]
    num_samples = sum(r["samples"] for r in rows)

    args.train_data = shards

    if getattr(args, "train_num_samples", None):
        _logger.info(
            "Manifest sample total (%d) ignored; --train-num-samples=%d given on the CLI.",
            num_samples,
            args.train_num_samples,
        )
    else:
        args.train_num_samples = num_samples

    resampled = bool(getattr(args, "dataset_resampled", False))

    # Manifest's own per-shard weights (None if raw+uniform).
    norm_source_weights = _normalized_source_weights(rows)
    manifest_per_shard = _per_shard(norm_source_weights, rows)
    manifest_uniform = len(set(round(w, 12) for w in manifest_per_shard)) <= 1
    manifest_weights = (
        None if (manifest.get("weight_mode", "raw") == "raw" and manifest_uniform) else manifest_per_shard
    )

    # CLI factors override manifest weights; expand the ::-string to per-shard so it matches the list train_data.
    cli_factors = getattr(args, "train_data_upsampling_factors", None)
    if isinstance(cli_factors, str):
        # Explicit CLI weights override the manifest's. They only mean anything under resampling -- reject early
        # here (with manifest context) rather than letting the run reach the downstream assert after logging a
        # misleading weighted-mix summary.
        if not resampled:
            raise ValueError(
                "--train-data-upsampling-factors requires --dataset-resampled (sampling with replacement)."
            )
        # Expand the ::-string to per-shard so it matches the list-valued train_data.
        applied = _parse_cli_upsampling_factors(cli_factors, source_shard_counts)
        applied_source_weights = _collapse_to_sources(applied, source_shard_counts)
        weight_origin = "CLI --train-data-upsampling-factors"
    elif manifest_weights is not None and not resampled:
        # Weights only take effect under shard resampling. Reading the spec in source order is a valid mode, so
        # just ignore the weights (size-proportional reading) and say so, rather than failing.
        _logger.info(
            "Manifest sampling weights ignored: not sampling with replacement (reading shards in "
            "order, size-proportional). Pass --dataset-resampled to apply them."
        )
        applied = None
        applied_source_weights = None
        weight_origin = "manifest"
    else:
        applied = manifest_weights
        applied_source_weights = norm_source_weights if manifest_weights is not None else None
        weight_origin = "manifest"

    args.train_data_upsampling_factors = applied  # per-shard list, or None for uniform / ignored

    _log_manifest_summary(rows, applied_source_weights, num_samples, resampled, weight_origin)


def _collapse_to_sources(per_shard: List[float], source_shard_counts: List[int]) -> List[float]:
    """Reduce a per-shard weight list back to one value per source (mean over the source's shards) for display."""
    out, i = [], 0
    for n in source_shard_counts:
        seg = per_shard[i:i + n]
        i += n
        out.append(sum(seg) / len(seg) if seg else 0.0)
    return out


def _log_manifest_summary(rows, applied_source_weights, num_samples, resampled, weight_origin) -> None:
    n_shards = sum(len(r["shards"]) for r in rows)
    _logger.info(
        "Loaded train-data manifest: %d sources, %d shards, %d samples, resampled=%s, weights=%s",
        len(rows),
        n_shards,
        num_samples,
        resampled,
        weight_origin if applied_source_weights is not None else "uniform",
    )
    # Effective sample-level mix: weight * samples when weights are applied, else proportional to samples.
    if applied_source_weights is not None:
        mass = [w * r["samples"] for w, r in zip(applied_source_weights, rows)]
    else:
        mass = [r["samples"] for r in rows]
    total = sum(mass) or 1.0
    for idx, r in enumerate(rows):
        w = applied_source_weights[idx] if applied_source_weights is not None else 1.0
        _logger.info(
            "  %-24s shards=%-6d samples=%-12d weight=%-8.4g ~mix=%.1f%%",
            r["name"],
            len(r["shards"]),
            r["samples"],
            w,
            100.0 * mass[idx] / total,
        )
