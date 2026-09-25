#!/usr/bin/env python3
"""
token_tracker.py: Static payload token profiler for A/B "value sprint" testing.

This utility estimates the token weight of one or more files or directories so a
"before" (raw) workflow can be compared against an "after" (optimized) workflow.
It is deliberately non-interactive so it can be embedded in a repeatable process
and run identically by multiple testers.

IMPORTANT: what this does and does NOT measure
----------------------------------------------
This tool counts tokens in the *text you point it at*, using a local tokenizer
(tiktoken). It does NOT read Cursor's own usage records, and it does not include
the system prompt, tool schemas, injected codebase context, or chat history that
a real Cursor request also carries. It is a payload-size estimator, not a meter
of what Cursor actually billed.

Tokenizer accuracy:
  * OpenAI models (gpt-4, gpt-4o, o-series, etc.) are tokenized with the matching
    tiktoken encoding, so those counts are exact for the given text.
  * Anthropic / Claude models have NO public offline tokenizer that reproduces
    current Claude token counts. When a Claude/Anthropic model is requested this
    tool falls back to an OpenAI encoding as a PROXY and flags every such count
    as an estimate (is_estimate=True). Treat Claude numbers as approximate.

Usage examples
--------------
  # Baseline the raw workflow, append a labeled row-set to a CSV
  python token_tracker.py --path ./data_before --label before --model gpt-4o --csv results.csv

  # Same run for the optimized payload
  python token_tracker.py --path ./data_after --label after --model gpt-4o --csv results.csv

  # Estimate Claude weight (clearly marked as an estimate) and also dump JSON
  python token_tracker.py --path ./data_after --model claude-sonnet --csv results.csv --json run.json

Offline / restricted environments (e.g. a customer VDI)
-------------------------------------------------------
tiktoken downloads its encoding tables from the internet the first time an
encoding is used. Where outbound traffic is blocked, pre-populate the cache on a
networked machine and carry it in:
  1) On a networked machine:
       export TIKTOKEN_CACHE_DIR=/some/dir/tiktoken_cache
       python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')"
  2) Copy /some/dir/tiktoken_cache into the restricted environment.
  3) There, set TIKTOKEN_CACHE_DIR to that directory before running this tool.
If the tables cannot be loaded, this tool exits with code 3 and prints these steps.

Exit codes: 0 success, 1 bad arguments / nothing to analyze, 2 tiktoken missing,
3 tokenizer data could not be loaded (offline without a populated cache).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fnmatch
import json
import os
import sys
import uuid

TOOL_VERSION = "0.2.0"


class EncodingUnavailable(RuntimeError):
    """Raised when the tokenizer data cannot be loaded (usually: offline and the
    encoding is not present in the local tiktoken cache)."""

    def __init__(self, name, cause):
        self.encoding_name = name
        self.cause = cause
        super().__init__(str(cause))

# Directory / path segments skipped by default. Matched on path *parts*, not as
# a loose substring of the whole path.
DEFAULT_EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".cursor", ".venv", "venv",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "dist", "build",
}

# Files this size or larger are recorded as skipped rather than loaded into
# memory and tokenized. Override with --max-bytes.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_encoding(name: str):
    """
    Load a tiktoken encoding by name, turning any load failure (most importantly
    a blocked/offline download of the BPE data file) into a clean
    EncodingUnavailable with actionable guidance instead of a raw traceback.
    """
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # network error, missing cache, corrupt data, etc.
        raise EncodingUnavailable(name, exc) from exc


def resolve_encoding(model: str):
    """
    Map a model name to a tiktoken encoding.

    Returns (encoding, encoding_name, is_estimate, note).

    is_estimate is True whenever the encoding is only a proxy for the requested
    model's real tokenizer (currently: any Anthropic/Claude model, and any model
    tiktoken does not recognize).
    """
    from tiktoken.model import encoding_name_for_model

    model_l = (model or "").lower().strip()

    # Anthropic / Claude: no faithful offline tokenizer. Use an OpenAI encoding
    # as a proxy and flag the result as an estimate.
    if "claude" in model_l or "anthropic" in model_l:
        return (
            _load_encoding("o200k_base"),
            "o200k_base",
            True,
            "Claude/Anthropic have no public offline tokenizer; o200k_base used "
            "as a proxy. Counts are estimates.",
        )

    # Resolve the exact encoding name for a known OpenAI model. This is a plain
    # dictionary lookup and does not touch the network.
    try:
        name = encoding_name_for_model(model_l)
        return (_load_encoding(name), name, False, "")
    except KeyError:
        pass

    # Heuristic fallback for newer OpenAI families tiktoken may not know by name.
    if any(tag in model_l for tag in ("gpt-4o", "gpt-4.1", "o1", "o3", "o4", "omni", "o200k")):
        return (_load_encoding("o200k_base"), "o200k_base", True,
                f"Unknown model '{model}'; assumed o200k_base.")

    # Last-resort default. cl100k_base covers gpt-4 / gpt-3.5 / text-embedding-3.
    is_est = model_l not in ("", "gpt-4", "gpt-3.5-turbo", "gpt-35-turbo")
    note = "" if not is_est else f"Unknown model '{model}'; defaulted to cl100k_base."
    return (_load_encoding("cl100k_base"), "cl100k_base", is_est, note)


def _is_excluded(path: str, root: str, exclude_globs) -> bool:
    """True if any path segment is a default-excluded dir, or the path matches an
    exclude glob (matched against both the full path and the path relative to root)."""
    parts = os.path.normpath(path).split(os.sep)
    if any(part in DEFAULT_EXCLUDE_DIRS for part in parts):
        return True
    rel = os.path.relpath(path, root)
    for pattern in exclude_globs:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(rel, pattern):
            return True
    return False


def _looks_binary(path: str, sniff_bytes: int = 8192) -> bool:
    """Cheap binary sniff: a NUL byte in the first chunk means 'not text'."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(sniff_bytes)
    except OSError:
        return False
    return b"\x00" in chunk


def _collect_files(target: str, recursive: bool, exclude_globs):
    """Yield candidate file paths under a file-or-directory target."""
    if os.path.isfile(target):
        yield target
        return
    if not os.path.isdir(target):
        return
    root = target
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # Prune excluded directories in place so os.walk doesn't descend them.
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
            for name in filenames:
                fp = os.path.join(dirpath, name)
                if not _is_excluded(fp, root, exclude_globs):
                    yield fp
    else:
        for name in sorted(os.listdir(root)):
            fp = os.path.join(root, name)
            if os.path.isfile(fp) and not _is_excluded(fp, root, exclude_globs):
                yield fp


def analyze(paths, model, label, recursive, exclude_globs, max_bytes):
    """
    Walk the given paths, count tokens per file, and return a structured result
    dict. Every file produces a row with an explicit status so nothing is
    silently dropped.
    """
    enc, enc_name, is_estimate, enc_note = resolve_encoding(model)
    run_id = uuid.uuid4().hex[:12]
    timestamp = _utc_now_iso()

    rows = []
    seen = set()
    for target in paths:
        base_root = target if os.path.isdir(target) else os.path.dirname(target) or "."
        for fp in _collect_files(target, recursive, exclude_globs):
            real = os.path.realpath(fp)
            if real in seen:  # de-dupe symlinks / overlapping paths
                continue
            seen.add(real)

            row = {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "label": label or "",
                "model": model,
                "encoding": enc_name,
                "is_estimate": is_estimate,
                "path": fp,
                "rel_path": os.path.relpath(fp, base_root),
                "size_bytes": None,
                "tokens": 0,
                "status": "ok",
                "detail": "",
            }

            try:
                size = os.path.getsize(fp)
                row["size_bytes"] = size
                if size >= max_bytes:
                    row["status"] = "skipped_large"
                    row["detail"] = f">= max_bytes ({max_bytes})"
                    rows.append(row)
                    continue
                if _looks_binary(fp):
                    row["status"] = "skipped_binary"
                    row["detail"] = "NUL byte detected in sniff"
                    rows.append(row)
                    continue
                with open(fp, "r", encoding="utf-8", errors="strict") as fh:
                    content = fh.read()
                row["tokens"] = len(enc.encode(content))
            except UnicodeDecodeError as exc:
                row["status"] = "skipped_decode_error"
                row["detail"] = str(exc)
            except OSError as exc:
                row["status"] = "error"
                row["detail"] = str(exc)
            rows.append(row)

    counted = [r for r in rows if r["status"] == "ok"]
    skipped = [r for r in rows if r["status"] != "ok"]
    total_tokens = sum(r["tokens"] for r in counted)
    total_bytes = sum(r["size_bytes"] or 0 for r in counted)

    return {
        "tool_version": TOOL_VERSION,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "label": label or "",
        "model": model,
        "encoding": enc_name,
        "is_estimate": is_estimate,
        "encoding_note": enc_note,
        "paths": [os.path.abspath(p) for p in paths],
        "files": rows,
        "summary": {
            "files_counted": len(counted),
            "files_skipped": len(skipped),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
        },
    }


CSV_FIELDS = [
    "run_id", "timestamp_utc", "label", "model", "encoding", "is_estimate",
    "path", "rel_path", "size_bytes", "tokens", "status", "detail",
]


def write_csv(result, csv_path):
    """Append per-file rows to a CSV, writing a header only when the file is new."""
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for row in result["files"]:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def write_json(result, json_path):
    """Write the full structured run to a JSON file (one run per file)."""
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


def print_report(result):
    s = result["summary"]
    est = "  [ESTIMATE: proxy tokenizer]" if result["is_estimate"] else ""
    print()
    print(f"=== Token Analysis Profile ==={est}")
    print(f"Run ID:    {result['run_id']}")
    print(f"When:      {result['timestamp_utc']}")
    print(f"Label:     {result['label'] or '(none)'}")
    print(f"Model:     {result['model']}  (encoding: {result['encoding']})")
    if result["encoding_note"]:
        print(f"Note:      {result['encoding_note']}")
    print("-" * 74)
    print(f"{'File':<52} | {'Status':<16} | {'Tokens':>10}")
    print("-" * 74)
    for row in result["files"]:
        name = row["rel_path"]
        if len(name) > 49:
            name = "..." + name[-46:]
        tok = f"{row['tokens']:,}" if row["status"] == "ok" else "-"
        print(f"{name:<52} | {row['status']:<16} | {tok:>10}")
    print("-" * 74)
    print(f"Files counted: {s['files_counted']}   skipped: {s['files_skipped']}")
    print(f"{'TOTAL TOKENS':<52} | {'':<16} | {s['total_tokens']:>10,}")
    if result["is_estimate"]:
        print("\n*** Claude/Anthropic counts are ESTIMATES (proxy tokenizer). ***")
    print()


def build_parser():
    p = argparse.ArgumentParser(
        prog="token_tracker.py",
        description="Estimate token weight of files/directories for A/B payload testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--path", "-p", action="append", dest="paths", metavar="PATH",
        help="File or directory to analyze. Repeat --path to add several.",
    )
    p.add_argument(
        "--model", "-m", default="gpt-4o",
        help="Target model name (default: gpt-4o). Claude/Anthropic models are "
             "counted with a proxy tokenizer and flagged as estimates.",
    )
    p.add_argument(
        "--label", "-l", default="",
        help="Free-form label for this run, e.g. 'before' or 'after'. Written to output.",
    )
    p.add_argument(
        "--csv", dest="csv_path", metavar="FILE",
        help="Append per-file rows to this CSV (created with a header if new).",
    )
    p.add_argument(
        "--json", dest="json_path", metavar="FILE",
        help="Write the full structured run to this JSON file.",
    )
    p.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="Do not descend into subdirectories (default: recursive).",
    )
    p.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB",
        help="Glob of paths to skip (repeatable), e.g. --exclude '*.min.js'.",
    )
    p.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES, metavar="N",
        help=f"Skip files >= N bytes (default: {DEFAULT_MAX_BYTES}).",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress the console table (still writes CSV/JSON).",
    )
    p.set_defaults(recursive=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        import tiktoken  # noqa: F401  (imported for the early, friendly error)
    except ImportError:
        print("Error: 'tiktoken' is not installed. Run: pip install tiktoken", file=sys.stderr)
        return 2

    if not args.paths:
        print("Error: no --path given. Example: --path ./data_before", file=sys.stderr)
        return 1

    missing = [p for p in args.paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: path not found: {p}", file=sys.stderr)
        return 1

    try:
        result = analyze(
            paths=args.paths,
            model=args.model,
            label=args.label,
            recursive=args.recursive,
            exclude_globs=args.exclude,
            max_bytes=args.max_bytes,
        )
    except EncodingUnavailable as exc:
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", "(not set)")
        print(
            "Error: could not load the tokenizer data for encoding "
            f"'{exc.encoding_name}'.\n"
            f"  Underlying cause: {exc.cause}\n\n"
            "tiktoken downloads its encoding tables from the internet on first "
            "use. In a restricted/offline environment (e.g. a customer VDI) that "
            "download is blocked, which is almost certainly what happened here.\n\n"
            "To run fully offline, pre-populate the tiktoken cache on a machine "
            "that has internet, then carry it into the restricted environment:\n"
            "  1) On a networked machine:\n"
            "       export TIKTOKEN_CACHE_DIR=/some/dir/tiktoken_cache\n"
            f"       python -c \"import tiktoken; tiktoken.get_encoding('{exc.encoding_name}')\"\n"
            "  2) Copy /some/dir/tiktoken_cache into the restricted environment.\n"
            "  3) There, set TIKTOKEN_CACHE_DIR to that copied directory and re-run.\n"
            f"  (TIKTOKEN_CACHE_DIR is currently: {cache_dir})",
            file=sys.stderr,
        )
        return 3

    if not args.quiet:
        print_report(result)
    if args.csv_path:
        write_csv(result, args.csv_path)
        if not args.quiet:
            print(f"Appended {len(result['files'])} row(s) to {args.csv_path}")
    if args.json_path:
        write_json(result, args.json_path)
        if not args.quiet:
            print(f"Wrote run JSON to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
