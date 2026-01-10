#!/usr/bin/env python3
"""
Parse optuna-iTransformer.rtf as-is and extract:
- trial_number
- kl_div_loss  (Optuna "value")
- all hyperparameters from "parameters: \{...\}"

Works even if the file is RTF-ish: it just searches for "Trial X finished with value: ..."
and tolerates the backslash-escaped braces and trailing "\".

Usage:
  python parse_optuna_as_is.py --input optuna-iTransformer.rtf --out_csv trials.csv --out_json trials.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


# Matches segments like:
# Trial 0 finished with value: 0.5879 and parameters: \{... \}. Best is trial ...
TRIAL_RE = re.compile(
    r"""
    Trial\s+(?P<trial>\d+)\s+finished\s+with\s+value:\s*
    (?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)
    \s+and\s+parameters:\s*
    (?P<params>\\?\{.*?\\?\})
    \.\s*Best\s+is\s+trial
    """,
    re.VERBOSE | re.DOTALL,
    )


def read_raw_text(path: Path) -> str:
    # Read bytes and decode permissively so we never crash on encoding junk.
    b = path.read_bytes()
    return b.decode("utf-8", errors="ignore")


def normalize_text(s: str) -> str:
    """
    Minimal normalization that does NOT destroy parameter braces.
    - Convert common RTF newlines to real newlines
    - Keep everything else intact
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace(r"\par", "\n")
    return s


def unescape_params_dict(raw: str) -> str:
    """
    Your file stores dict braces as \{ and \} (see the log lines).
    We convert \{ -> { and \} -> } so ast.literal_eval can parse.
    Also trim trailing backslashes that appear at the end of lines.
    """
    s = raw.strip()
    # Strip trailing "\" (line continuation artifacts)
    s = re.sub(r"\\+\s*$", "", s)
    # Unescape braces
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    return s


def parse_trials(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for m in TRIAL_RE.finditer(text):
        trial_num = int(m.group("trial"))
        value = float(m.group("value"))
        params_raw = m.group("params")

        params_str = unescape_params_dict(params_raw)

        try:
            params = ast.literal_eval(params_str)
            if not isinstance(params, dict):
                raise ValueError(f"parameters is not a dict (got {type(params)})")
        except Exception as e:
            # Keep going; store raw for debugging
            params = {"_params_parse_error": str(e), "_params_raw": params_str}

        row: Dict[str, Any] = {
            "trial_number": trial_num,
            "kl_div_loss": value,
        }
        for k, v in params.items():
            row[str(k)] = v
        rows.append(row)

    rows.sort(key=lambda r: r["trial_number"])
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    # Wide columns: union of keys over all rows
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)

    # Put the main fields first
    for key in ["trial_number", "kl_div_loss"]:
        if key in cols:
            cols.remove(key)
    cols = ["trial_number", "kl_div_loss"] + cols

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="optuna-iTransformer.rtf")
    ap.add_argument("--out_csv", default="optuna_trials.csv")
    ap.add_argument("--out_json", default="optuna_trials.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    raw = read_raw_text(in_path)
    text = normalize_text(raw)

    rows = parse_trials(text)
    if not rows:
        raise SystemExit(
            "Parsed 0 trials. This parser expects lines like:\n"
            "Trial N finished with value: X and parameters: \\{...\\}. Best is trial ...\n"
            "If your file differs, paste 2 raw lines around a Trial entry."
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_csv(rows, out_csv)
    write_json(rows, out_json)

    best = min(rows, key=lambda r: r["kl_div_loss"])
    print(f"Parsed {len(rows)} trials from {in_path}")
    print(f"Best: trial={best['trial_number']} kl_div_loss={best['kl_div_loss']}")
    print(f"Wrote: {out_csv.resolve()}")
    print(f"Wrote: {out_json.resolve()}")


if __name__ == "__main__":
    main()