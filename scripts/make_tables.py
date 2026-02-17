#!/usr/bin/env python3
"""
make_tables.py
==============
Read metrics_summary.json and produce:
  - tables/results.csv
  - tables/results.tex   (booktabs, ACL/EMNLP-friendly)
  - qualitative.md        (10-20 examples with diff, prob shift, bucket)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUCKET_NAMES = {
    "A": "intrinsic→no",
    "B": "extrinsic→no",
    "C": "no→extrinsic",
    "D": "no→intrinsic",
}


def _pct(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{digits}f}"


def _f(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _text_diff(original: str, counterfactual: str) -> str:
    """Return a unified diff string (compact)."""
    a = original.split()
    b = counterfactual.split()
    sm = difflib.SequenceMatcher(None, a, b)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            snippet = " ".join(a[i1:i2])
            if len(snippet) > 60:
                snippet = snippet[:30] + " … " + snippet[-30:]
            parts.append(snippet)
        elif tag == "replace":
            parts.append(f"~~{' '.join(a[i1:i2])}~~ → **{' '.join(b[j1:j2])}**")
        elif tag == "delete":
            parts.append(f"~~{' '.join(a[i1:i2])}~~")
        elif tag == "insert":
            parts.append(f"**+{' '.join(b[j1:j2])}**")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def make_standard_table(metrics: Dict[str, Any]) -> List[List[str]]:
    """Build rows for standard metrics across models."""
    header = ["Model", "Acc", "F1-Macro", "F1-Wt", "F1-no", "F1-intr", "F1-extr"]
    rows = [header]
    for model_name, m in sorted(metrics.items()):
        s = m.get("standard", {})
        pc = s.get("per_class", {})
        rows.append([
            model_name,
            _pct(s.get("accuracy")),
            _pct(s.get("f1_macro")),
            _pct(s.get("f1_weighted")),
            _pct(pc.get("no", {}).get("f1")),
            _pct(pc.get("intrinsic", {}).get("f1")),
            _pct(pc.get("extrinsic", {}).get("f1")),
        ])
    return rows


def make_cf_table(metrics: Dict[str, Any]) -> List[List[str]]:
    """Build rows for counterfactual metrics."""
    header = ["Model", "CFR", "DCS", "ΔConf", "MESR@1", "MESR@3", "MESR@5", "MESR@10", "Coverage"]
    rows = [header]
    for model_name, m in sorted(metrics.items()):
        c = m.get("counterfactual", {})
        if not c:
            continue
        rows.append([
            model_name,
            _pct(c.get("CFR")),
            _pct(c.get("DCS")),
            _f(c.get("delta_conf_mean"), 3),
            _pct(c.get("MESR@1")),
            _pct(c.get("MESR@3")),
            _pct(c.get("MESR@5")),
            _pct(c.get("MESR@10")),
            _pct(c.get("coverage_rate")),
        ])
    return rows


def make_bucket_table(metrics: Dict[str, Any]) -> List[List[str]]:
    """Per-bucket CFR/DCS table."""
    header = ["Model", "Bucket", "Direction", "CFR", "DCS", "ΔConf", "n_valid"]
    rows = [header]
    for model_name, m in sorted(metrics.items()):
        bk = m.get("counterfactual_buckets", {})
        for b in ["A", "B", "C", "D"]:
            bm = bk.get(b, {})
            if not bm:
                continue
            rows.append([
                model_name,
                b,
                BUCKET_NAMES[b],
                _pct(bm.get("CFR")),
                _pct(bm.get("DCS")),
                _f(bm.get("delta_conf_mean"), 3),
                str(bm.get("n_valid", 0)),
            ])
    return rows


def make_stability_table(metrics: Dict[str, Any]) -> List[List[str]]:
    header = ["Model", "StabScore", "FlipRate", "SensGap"]
    rows = [header]
    for model_name, m in sorted(metrics.items()):
        st = m.get("stability", {})
        if not st:
            continue
        rows.append([
            model_name,
            _f(st.get("stability_score")),
            _f(st.get("flip_rate_nonsemantic")),
            _f(st.get("sensitivity_gap")),
        ])
    return rows


def make_ablation_table(metrics: Dict[str, Any]) -> List[List[str]]:
    header = ["Model", "NoCtx-Acc", "NoCtx-F1", "NoCtx-CFR", "CFR-Drop-A", "CFR-Drop-B", "CFR-Drop-C", "CFR-Drop-D"]
    rows = [header]
    for model_name, m in sorted(metrics.items()):
        abl = m.get("ablation", {})
        if not abl:
            continue
        s = abl.get("standard_no_context", {})
        cfc = abl.get("cf_no_context", {})
        drops = abl.get("cfr_drop_per_bucket", {})
        rows.append([
            model_name,
            _pct(s.get("accuracy")),
            _pct(s.get("f1_macro")),
            _pct(cfc.get("CFR")),
            _f(drops.get("A"), 3),
            _f(drops.get("B"), 3),
            _f(drops.get("C"), 3),
            _f(drops.get("D"), 3),
        ])
    return rows


# ---------------------------------------------------------------------------
# CSV / LaTeX writers
# ---------------------------------------------------------------------------


def write_csv(tables: Dict[str, List[List[str]]], out_path: Path):
    """Write all tables into one CSV with section separators."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for name, rows in tables.items():
            writer.writerow([f"=== {name} ==="])
            for row in rows:
                writer.writerow(row)
            writer.writerow([])


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def rows_to_latex(table_name: str, rows: List[List[str]], caption: str = "") -> str:
    """Convert a table (list of rows) to a LaTeX booktabs table."""
    if not rows or len(rows) < 2:
        return ""
    header = rows[0]
    data = rows[1:]
    ncols = len(header)
    col_fmt = "l" + "r" * (ncols - 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    if caption:
        lines.append(r"\caption{" + _latex_escape(caption) + "}")
    lines.append(r"\label{tab:" + table_name.lower().replace(" ", "_") + "}")
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in header) + r" \\")
    lines.append(r"\midrule")
    for row in data:
        lines.append(" & ".join(_latex_escape(c) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------


def build_qualitative(
    cf_path: Path,
    preds_cf_dir: Path,
    n_examples: int = 20,
) -> str:
    """Sample qualitative examples and build markdown."""
    if not cf_path.exists():
        return "# Qualitative Examples\n\nNo counterfactual pairs available.\n"

    # Load CF pairs
    cf_pairs = []
    with open(cf_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cf_pairs.append(json.loads(line))

    # Load predictions (pick first model found)
    pred_files = sorted(preds_cf_dir.glob("preds_counterfactual_*.jsonl"))
    preds_map: Dict[str, Dict] = {}
    model_used = "unknown"
    if pred_files:
        model_used = pred_files[0].stem.replace("preds_counterfactual_", "")
        with open(pred_files[0], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    key = f"{obj['id']}_{obj['bucket']}"
                    preds_map[key] = obj

    # Select interesting examples: prefer cases with large prob shifts or mismatches
    scored = []
    for cp in cf_pairs:
        if not cp.get("valid_cf", False):
            continue
        cf_resp = cp.get("counterfactual_response", "")
        if not cf_resp:
            continue

        key = f"{cp['id']}_{cp['bucket']}"
        pred_info = preds_map.get(key, {})
        probs_cf = pred_info.get("probs_cf", [0, 0, 0])
        probs_orig = pred_info.get("probs_orig", [0, 0, 0])
        pred_cf = pred_info.get("pred_cf", -1)

        # Score: magnitude of prob shift
        shift = sum(abs(a - b) for a, b in zip(probs_orig, probs_cf))
        scored.append((shift, cp, pred_info))

    # Sort by shift desc, take top n
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:n_examples]

    # Build markdown
    id2label = {0: "no", 1: "intrinsic", 2: "extrinsic"}
    lines = [
        f"# Qualitative Counterfactual Examples",
        f"",
        f"Model: **{model_used}**",
        f"",
        f"Selected {len(selected)} examples with largest probability shifts.",
        "",
    ]

    for idx, (shift, cp, pred) in enumerate(selected, 1):
        bucket = cp.get("bucket", "?")
        direction = BUCKET_NAMES.get(bucket, "?")
        resp_orig = cp.get("response", "")
        resp_cf = cp.get("counterfactual_response", "")
        context_snip = _truncate(cp.get("context", ""), 200)
        prompt_snip = _truncate(cp.get("prompt", ""), 150)

        probs_orig = pred.get("probs_orig", [0, 0, 0])
        probs_cf = pred.get("probs_cf", [0, 0, 0])
        pred_orig = pred.get("pred_orig", -1)
        pred_cf = pred.get("pred_cf", -1)
        expected = cp.get("y_cf_expected", "?")

        diff_str = _text_diff(resp_orig, resp_cf)

        lines.append(f"---")
        lines.append(f"## Example {idx}  |  Bucket {bucket} ({direction})")
        lines.append(f"")
        lines.append(f"**ID:** `{cp.get('id', '?')}`")
        lines.append(f"")
        lines.append(f"**Context (truncated):** {context_snip}")
        lines.append(f"")
        lines.append(f"**Prompt:** {prompt_snip}")
        lines.append(f"")
        lines.append(f"**Original Response:** {_truncate(resp_orig, 300)}")
        lines.append(f"")
        lines.append(f"**Counterfactual Response:** {_truncate(resp_cf, 300)}")
        lines.append(f"")
        lines.append(f"**Diff:** {diff_str}")
        lines.append(f"")
        lines.append(f"| Metric | Original | Counterfactual |")
        lines.append(f"|--------|----------|----------------|")
        lines.append(f"| Prediction | {id2label.get(pred_orig, '?')} | {id2label.get(pred_cf, '?')} |")
        lines.append(f"| Expected | {cp.get('y_true', '?')} | {expected} |")
        lines.append(f"| P(no) | {probs_orig[0]:.4f} | {probs_cf[0]:.4f} |")
        lines.append(f"| P(intrinsic) | {probs_orig[1]:.4f} | {probs_cf[1]:.4f} |")
        lines.append(f"| P(extrinsic) | {probs_orig[2]:.4f} | {probs_cf[2]:.4f} |")
        lines.append(f"| Token edits | {cp.get('edit_stats', {}).get('token_edits', '?')} | — |")
        lines.append(f"| Prob shift Σ|Δp| | {shift:.4f} | — |")
        lines.append(f"")
        rationale = cp.get("rationale_short", "")
        if rationale:
            lines.append(f"**Rationale:** {rationale}")
            lines.append(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and qualitative analysis.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_qualitative", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "exp_out"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_summary.json"
    if not metrics_path.exists():
        print(f"[tables] ERROR: {metrics_path} not found. Run run_eval.py first.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"[tables] Loaded metrics for {len(metrics)} model(s).")

    # ---- Build tables ----
    all_tables = {}
    std = make_standard_table(metrics)
    all_tables["Standard Metrics"] = std

    cf = make_cf_table(metrics)
    if len(cf) > 1:
        all_tables["Counterfactual Metrics"] = cf

    bk = make_bucket_table(metrics)
    if len(bk) > 1:
        all_tables["Bucket-wise Metrics"] = bk

    stab = make_stability_table(metrics)
    if len(stab) > 1:
        all_tables["Stability"] = stab

    abl = make_ablation_table(metrics)
    if len(abl) > 1:
        all_tables["Ablation (No Context)"] = abl

    # ---- Write CSV ----
    csv_path = tables_dir / "results.csv"
    write_csv(all_tables, csv_path)
    print(f"[tables] CSV written to {csv_path}")

    # ---- Write LaTeX ----
    tex_path = tables_dir / "results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated tables for hallucination detection counterfactual evaluation\n")
        f.write("% Requires: \\usepackage{booktabs}\n\n")
        captions = {
            "Standard Metrics": "Standard evaluation metrics on the original test set.",
            "Counterfactual Metrics": "Counterfactual evaluation metrics (valid pairs only).",
            "Bucket-wise Metrics": "Per-bucket counterfactual metrics.",
            "Stability": "Stability evaluation with non-semantic edits.",
            "Ablation (No Context)": "No-context ablation results.",
        }
        for name, rows in all_tables.items():
            tex = rows_to_latex(name, rows, caption=captions.get(name, name))
            if tex:
                f.write(tex + "\n\n")
    print(f"[tables] LaTeX written to {tex_path}")

    # ---- Qualitative ----
    cf_path = out_dir / "counterfactual_pairs.jsonl"
    qual_md = build_qualitative(cf_path, out_dir, n_examples=args.n_qualitative)
    qual_path = out_dir / "qualitative.md"
    with open(qual_path, "w", encoding="utf-8") as f:
        f.write(qual_md)
    print(f"[tables] Qualitative markdown written to {qual_path}")

    print("[tables] Done.")


if __name__ == "__main__":
    main()
