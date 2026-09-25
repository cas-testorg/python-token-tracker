# python-token-tracker

A small, self-contained utility for measuring the token weight of files so a
"before" (raw) workflow can be compared against an "after" (optimized) workflow.
It is built for repeatable A/B "value sprint" testing, where the same measurement
has to be run identically by more than one person and the numbers have to hold up
to scrutiny.

## What this measures, and what it does not

Be precise about this up front, because it is the first thing a technical
reviewer will ask.

This tool counts the tokens in the text you point it at, using a local tokenizer
(tiktoken). That is a payload-size measurement. It is useful for showing how much
smaller a payload becomes after optimization.

It does not read Cursor's own usage records, and it does not include the system
prompt, tool schemas, injected codebase context, or conversation history that a
real Cursor request also carries. So the token totals here are not the same as
what Cursor actually billed for a session. Treat this as "how heavy is this
payload" evidence, not "here is our exact metered usage" evidence.

### Tokenizer accuracy

* OpenAI models (gpt-4, gpt-4o, the o-series, and so on) are tokenized with the
  matching tiktoken encoding, so those counts are exact for the given text.
* Anthropic / Claude models have no public offline tokenizer that reproduces
  current Claude token counts. When you request a Claude model, this tool uses an
  OpenAI encoding (o200k_base) as a proxy and flags every such run as an estimate
  (`is_estimate=True`). Lean on that flag when you present numbers, and prefer an
  OpenAI model for the "exact" story.

## Install

The tool needs Python 3.8+ and the `tiktoken` package. On macOS with a Homebrew
Python, the system interpreter is externally managed, so install into a
virtualenv rather than the system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tiktoken
```

## Usage

The tool is a non-interactive CLI so it can be embedded in a repeatable process.

```
python token_tracker.py --path PATH [--path PATH ...] [options]
```

Options:

| Flag | Meaning |
| :--- | :--- |
| `--path`, `-p` | File or directory to analyze. Repeat to add several. |
| `--model`, `-m` | Target model name (default `gpt-4o`). Claude models are proxied and flagged as estimates. |
| `--label`, `-l` | Free-form label for the run, for example `before` or `after`. Written to output. |
| `--csv FILE` | Append per-file rows to this CSV (a header is written only when the file is new). |
| `--json FILE` | Write the full structured run to this JSON file. |
| `--exclude GLOB` | Skip paths matching this glob. Repeatable, for example `--exclude '*.min.js'`. |
| `--max-bytes N` | Skip files at or above N bytes (default 10 MB). |
| `--no-recursive` | Do not descend into subdirectories. |
| `--quiet`, `-q` | Suppress the console table (still writes CSV/JSON). |

### A/B example

Baseline the raw payload, then measure the optimized payload into the same CSV:

```bash
python token_tracker.py --path ./data_before --label before --model gpt-4o --csv results.csv
python token_tracker.py --path ./data_after  --label after  --model gpt-4o --csv results.csv
```

Estimate the Claude weight of the optimized payload (clearly marked as an
estimate) and also capture a full JSON run:

```bash
python token_tracker.py --path ./data_after --model claude-sonnet --csv results.csv --json after.json
```

## Output

### CSV (appendable, one row per file)

Columns: `run_id`, `timestamp_utc`, `label`, `model`, `encoding`, `is_estimate`,
`path`, `rel_path`, `size_bytes`, `tokens`, `status`, `detail`.

Every file that is walked produces a row, including files that were skipped, so
nothing is dropped silently. The `status` column is one of:

* `ok` (counted),
* `skipped_binary` (a NUL byte was detected),
* `skipped_large` (at or above `--max-bytes`),
* `skipped_decode_error` (not valid UTF-8),
* `error` (could not be read).

When you aggregate the CSV, filter on `status == "ok"` before summing or
averaging. Skipped rows carry `tokens=0`, so counting them would understate your
per-file averages.

### JSON (one run per file)

A single run object with `tool_version`, `run_id`, `timestamp_utc`, `label`,
`model`, `encoding`, `is_estimate`, the resolved `paths`, the full per-file
`files` list, and a `summary` block (`files_counted`, `files_skipped`,
`total_tokens`, `total_bytes`).

## Offline and restricted environments (for example a customer VDI)

This matters for locked-down environments. tiktoken downloads its encoding tables
from the internet the first time an encoding is used. Where outbound traffic is
blocked, that download fails, and the tool exits with code 3 and prints these
steps rather than a stack trace.

To run fully offline, pre-populate the tiktoken cache on a machine that has
internet, then carry the cache into the restricted environment:

```bash
# 1) On a networked machine:
export TIKTOKEN_CACHE_DIR=/some/dir/tiktoken_cache
python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')"

# 2) Copy /some/dir/tiktoken_cache into the restricted environment.

# 3) There, set TIKTOKEN_CACHE_DIR to that directory before running the tool:
export TIKTOKEN_CACHE_DIR=/path/to/copied/tiktoken_cache
```

## Exit codes

| Code | Meaning |
| :--- | :--- |
| 0 | Success. |
| 1 | Bad arguments, or nothing to analyze. |
| 2 | `tiktoken` is not installed. |
| 3 | Tokenizer data could not be loaded (offline without a populated cache). |
