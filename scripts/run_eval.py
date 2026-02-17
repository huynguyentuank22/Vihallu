#!/usr/bin/env python3
"""
run_eval.py
===========
Evaluate a fine-tuned classifier (Qwen + LoRA) on:
  1) Original test set  -> Accuracy, Macro-F1, per-class F1
  2) Counterfactual set -> CFR, DCS, ΔConf, MESR@k, bucket-wise
  3) Stability (optional) -> non-semantic edit flip rate, SS, SG
  4) No-context ablation -> all counterfactual metrics with Context=""

Outputs:
  - preds_original_{model}.jsonl
  - preds_counterfactual_{model}.jsonl
  - metrics_summary.json

Usage: edit CONFIG section below, then run `python run_eval.py`.
"""

from __future__ import annotations
# !pip install -q bitsandbytes
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================= CONFIG =====================================
# Paths to fine-tuned model dirs (relative to project root or absolute).
# Add / uncomment entries as needed.  If empty → error.
MODEL_PATHS: List[str] = [
    "/kaggle/input/models/huy281204/detect-hallu/transformers/default/1/Qwen_Qwen3_1.7B",
    "/kaggle/input/models/huy281204/detect-hallu/transformers/default/1/Qwen_Qwen3_4B",
    "/kaggle/input/models/huy281204/detect-hallu/transformers/default/1/Qwen_Qwen3_8B"
]

TEST_PATH: str = "/kaggle/input/datasets/huy281204/vihallu/data/vihallu_test.csv"
COUNTERFACTUAL_PATH: str = "/kaggle/input/datasets/huy281204/counterfactual-data/counterfactual_pairs.jsonl"
OUT_DIR: str = "/kaggle/working/"

# If a shared label_mappings.json exists outside each model dir, set here.
# Otherwise leave None → each model dir's own label_mappings.json is used.
LABEL_MAPPING_PATH: Optional[str] = None

BATCH_SIZE: int = 16
DEVICE: str = "cuda"  # "cuda" or "cpu"

DO_STABILITY: bool = True
DO_NO_CONTEXT: bool = True
STABILITY_N: int = 200   # max samples for stability eval (0 = all)

# Pre-computed results (set to skip re-inference on test set)
# CSV with columns: Model, Accuracy, F1 Macro, ...  (optional)
MODEL_COMPARISON_CSV: Optional[str] = None  # e.g. "results/model_comparison.csv"

SEED: int = 42
# ========================= END CONFIG ====================================

# ---------------------------------------------------------------------------
# Default label mapping (used when label_mappings.json is absent)
# ---------------------------------------------------------------------------
DEFAULT_LABEL2ID: Dict[str, int] = {"intrinsic": 0, "extrinsic": 1, "no": 2}
DEFAULT_ID2LABEL: Dict[int, str] = {v: k for k, v in DEFAULT_LABEL2ID.items()}

# Model name normalisation: CSV uses "Qwen3-8B", dirs use "Qwen_Qwen3_8B"
_MODEL_NAME_MAP: Dict[str, str] = {
    "Qwen2.5-0.5B": "Qwen_Qwen2.5_0.5B",
    "Qwen2.5-1.5B": "Qwen_Qwen2.5_1.5B",
    "Qwen2.5-3B":   "Qwen_Qwen2.5_3B",
    "Qwen2.5-7B":   "Qwen_Qwen2.5_7B",
    "Qwen3-0.6B":   "Qwen_Qwen3_0.6B",
    "Qwen3-1.7B":   "Qwen_Qwen3_1.7B",
    "Qwen3-4B":     "Qwen_Qwen3_4B",
    "Qwen3-8B":     "Qwen_Qwen3_8B",
}


def _load_csv_metrics(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load pre-computed standard metrics from model_comparison.csv.

    Returns {dir_name: {accuracy, f1_macro, f1_weighted, ...}}.
    """
    df = pd.read_csv(csv_path)
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        csv_name = str(row["Model"]).strip()
        dir_name = _MODEL_NAME_MAP.get(csv_name, csv_name)
        out[dir_name] = {
            "accuracy": float(row.get("Accuracy", 0)),
            "f1_macro": float(row.get("F1 Macro", 0)),
            "f1_weighted": float(row.get("F1 Weighted", 0)),
        }
    return out


def _load_cached_preds(
    model_dir: Path,
    df_test: pd.DataFrame,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load per-sample predictions from test_predictions.csv inside model_dir.

    The CSV must have columns: predicted_label, prob_no, prob_intrinsic, prob_extrinsic.
    Matching is done by **positional index** (row order) because the test CSV
    ids are UUIDs while test_predictions.csv ids are sequential integers (0,1,2,...).
    Returns (orig_preds, orig_probs) arrays aligned with df_test row order,
    or None if the file doesn't exist or row counts don't match.
    """
    cache_path = model_dir / "test_predictions.csv"
    if not cache_path.exists():
        return None

    df_cache = pd.read_csv(cache_path)

    n = len(df_test)
    if len(df_cache) != n:
        print(f"  [cache] Row count mismatch: test={n}, cache={len(df_cache)} → re-running inference")
        return None

    num_labels = len(id2label)

    # Prob columns: prob_<label_name>
    prob_cols = {lid: f"prob_{id2label[lid]}" for lid in range(num_labels)}

    orig_preds = np.zeros(n, dtype=np.int64)
    orig_probs = np.zeros((n, num_labels), dtype=np.float32)

    for idx in range(n):
        row = df_cache.iloc[idx]
        pred_label = str(row["predicted_label"]).strip()
        orig_preds[idx] = label2id.get(pred_label, 0)
        for lid in range(num_labels):
            col = prob_cols[lid]
            orig_probs[idx, lid] = float(row.get(col, 0.0))

    print(f"  [cache] Loaded {n} predictions from {cache_path}")
    return orig_preds, orig_probs

# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------


def load_classifier(model_dir: str, device: str = "cuda"):
    """Load a fine-tuned Qwen + LoRA classifier from *model_dir*.

    model_dir must contain:
      - training_config.json  (with model_id)
      - label_mappings.json
      - adapter_config.json / adapter_model.safetensors
      - tokenizer files
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    # Load configs
    with open(os.path.join(model_dir, "training_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(model_dir, "label_mappings.json"), "r", encoding="utf-8") as f:
        lm = json.load(f)

    model_id = cfg["model_id"]
    label2id = {k: int(v) for k, v in lm["label2id"].items()}
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        id2label=id2label,
        label2id=label2id,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    # Ensure model config knows the pad token (needed for batched inference)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if base.config.pad_token_id is None:
        base.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, label2id, id2label


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _make_input(context: str, prompt: str, response: str) -> str:
        return f"""
Context: {context}
Prompt: {prompt}
Response: {response}
"""


@torch.no_grad()
def predict_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    desc: str = "Predicting",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred_ids, probs) arrays for a list of input texts."""
    all_preds = []
    all_probs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.append(preds)
        all_probs.append(probs)
        done = min(i + batch_size, len(texts))
        print(f"\r  [{desc}] {done}/{len(texts)} ({batch_idx+1}/{n_batches} batches)", end="", flush=True)
    print()  # newline after progress
    return np.concatenate(all_preds), np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Label-order helper
# ---------------------------------------------------------------------------


def get_label_order(id2label: Dict[int, str]) -> Tuple[List[int], List[str]]:
    """Return (sorted_label_ids, corresponding_names) from id2label."""
    labels = sorted(id2label.keys())
    names = [id2label[i] for i in labels]
    return labels, names


# ---------------------------------------------------------------------------
# Edit-size helper (unified across bucket types)
# ---------------------------------------------------------------------------


def _to_int(x) -> int:
    """Robust cast to int; handles str '16.0', ' 16', None, etc."""
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 10**9


def get_edit_size(edit_stats) -> int:
    """Extract a single edit-size integer from *edit_stats*.

    Priority:
      1) edit_tokens_total  (if pipeline wrote a combined field)
      2) token_edits        (bucket A / C / D)
      3) deleted_tokens     (bucket B)
      4) fallback → very large number (no data)
    """
    if not isinstance(edit_stats, dict):
        return 10**9
    for key in ("edit_tokens_total", "token_edits", "deleted_tokens"):
        if key in edit_stats:
            return _to_int(edit_stats[key])
    return 10**9


def compute_edit_size_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute edit_size distribution statistics over valid CF records."""
    valid = [r for r in records if r.get("valid_cf", False)]
    if not valid:
        return {}
    sizes = [r["edit_size"] for r in valid]
    finite = [s for s in sizes if s < 10**9]
    missing = len(sizes) - len(finite)
    n_le = {}
    for k in [1, 3, 5, 10]:
        n_le[f"n_le_{k}"] = sum(1 for s in sizes if s <= k)
    stats: Dict[str, Any] = {
        "n_valid": len(valid),
        "missing_count": missing,
        **n_le,
    }
    if finite:
        arr = np.array(finite, dtype=np.float64)
        stats["min"] = int(np.min(arr))
        stats["median"] = float(np.median(arr))
        stats["p90"] = float(np.percentile(arr, 90))
        stats["max"] = int(np.max(arr))
    return stats


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------


def compute_standard_metrics(
    y_true: List[int],
    y_pred: List[int],
    probs: np.ndarray,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """Compute accuracy, macro-F1, weighted-F1, per-class F1, confusion matrix.

    Label order is derived from *id2label* so it is always consistent.
    """
    labels, names = get_label_order(id2label)

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = {}
    for name in names:
        if name in report:
            per_class[name] = {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wt),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Counterfactual metrics
# ---------------------------------------------------------------------------


def compute_cf_metrics(
    records: List[Dict[str, Any]],
    label2id: Dict[str, int],
) -> Dict[str, Any]:
    """Compute CFR, DCS, ΔConf, MESR@k from counterfactual prediction records.

    Each record must have:
      - y_true (str)         : original ground-truth label
      - y_cf_expected (str)  : expected label after CF
      - pred_orig (int)      : classifier prediction on original
      - pred_cf (int)        : classifier prediction on CF
      - probs_orig (list)    : probabilities on original (len=3)
      - probs_cf (list)      : probabilities on CF (len=3)
      - edit_size (int)      : unified edit distance (via get_edit_size)
      - valid_cf (bool)
    """
    valid = [r for r in records if r.get("valid_cf", False)]
    if not valid:
        return {"n_valid": 0}

    n = len(valid)
    # CFR: fraction where CF prediction == expected label
    cfr_hits = sum(
        1 for r in valid if r["pred_cf"] == label2id[r["y_cf_expected"]]
    )
    cfr = cfr_hits / n

    # DCS: among samples where pred flipped (pred_cf != pred_orig), what frac flipped to expected?
    flips = [r for r in valid if r["pred_cf"] != r["pred_orig"]]
    dcs = 0.0
    if flips:
        dcs_hits = sum(1 for r in flips if r["pred_cf"] == label2id[r["y_cf_expected"]])
        dcs = dcs_hits / len(flips)

    # ΔConf = p(expected_cf) on CF  -  p(orig_class) on original
    delta_conf_list = []
    for r in valid:
        expected_id = label2id[r["y_cf_expected"]]
        p_expected_cf = r["probs_cf"][expected_id]
        p_orig_class = r["probs_orig"][r["pred_orig"]]
        delta_conf_list.append(p_expected_cf - p_orig_class)
    delta_conf_mean = float(np.mean(delta_conf_list))
    delta_conf_median = float(np.median(delta_conf_list))

    # MESR@k: among samples with edit_size <= k, fraction where pred_cf == expected
    mesr = {}
    mesr_counts = {}
    for k in [1, 3, 5, 10]:
        subset = [r for r in valid if r.get("edit_size", 10**9) <= k]
        mesr_counts[f"n_le_{k}"] = len(subset)
        if subset:
            hits = sum(1 for r in subset if r["pred_cf"] == label2id[r["y_cf_expected"]])
            mesr[f"MESR@{k}"] = hits / len(subset)
        else:
            mesr[f"MESR@{k}"] = None

    # edit_size distribution stats (for sanity / paper appendix)
    es_stats = compute_edit_size_stats(records)

    return {
        "n_valid": n,
        "n_total": len(records),
        "coverage_rate": n / max(len(records), 1),
        "CFR": cfr,
        "DCS": dcs,
        "n_flips": len(flips),
        "delta_conf_mean": delta_conf_mean,
        "delta_conf_median": delta_conf_median,
        **mesr,
        "mesr_subset_counts": mesr_counts,
        "edit_size_stats": es_stats,
    }


def compute_bucket_metrics(
    records: List[Dict[str, Any]],
    label2id: Dict[str, int],
) -> Dict[str, Any]:
    """Group records by bucket and compute per-bucket CF metrics."""
    buckets: Dict[str, list] = defaultdict(list)
    for r in records:
        buckets[r["bucket"]].append(r)
    result = {}
    for bk in sorted(buckets.keys()):
        result[bk] = compute_cf_metrics(buckets[bk], label2id)
    return result


# ---------------------------------------------------------------------------
# Stability (non-semantic edits)
# ---------------------------------------------------------------------------


def generate_nonsemantic_edits(
    context: str, prompt: str, response: str
) -> Optional[str]:
    """Create a non-semantic variant of the input (punctuation / whitespace / sentence order).
    Returns None if no edit can be made.
    """
    sents = re.split(r"(?<=[.?!。])\s+", response.strip())
    edited = response
    changed = False

    # 1) swap sentence order if 2+
    if len(sents) >= 2:
        new_sents = sents[:]
        new_sents[0], new_sents[1] = new_sents[1], new_sents[0]
        edited = " ".join(new_sents)
        changed = True
    else:
        # 2) normalise punctuation (double space -> single, add/remove trailing period)
        if edited.endswith("."):
            edited = edited[:-1]
        else:
            edited = edited + "."
        changed = True

    if not changed:
        return None
    return edited


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ---- reproducibility ----
    set_seed(SEED)

    # Resolve project root: works both as a script (__file__) and in notebooks
    try:
        project_root = Path(__file__).resolve().parent.parent
    except NameError:
        project_root = Path.cwd()  # fallback for Jupyter / Kaggle

    # ---- resolve paths (absolute paths in CONFIG are used as-is) ----
    def _resolve(p_str: str) -> Path:
        p = Path(p_str)
        return p if p.is_absolute() else project_root / p

    test_path = _resolve(TEST_PATH)
    out_dir = _resolve(OUT_DIR)
    cf_path = _resolve(COUNTERFACTUAL_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve model dirs ----
    if not MODEL_PATHS:
        raise ValueError("[run_eval] MODEL_PATHS is empty. Set at least one model path in the CONFIG section.")

    model_dirs: List[Path] = []
    for mp in MODEL_PATHS:
        p = _resolve(mp)
        if not p.exists():
            print(f"[run_eval] WARNING: model path does not exist: {p}")
        model_dirs.append(p)

    print(f"[eval] Test path : {test_path}")
    print(f"[eval] CF path   : {cf_path}")
    print(f"[eval] Out dir   : {out_dir}")
    print(f"[eval] Models    : {[d.name for d in model_dirs]}")
    print(f"[eval] Seed      : {SEED}")

    # ---- load test data ----
    df_test = pd.read_csv(test_path)
    print(f"[eval] Test samples: {len(df_test)}")

    # ---- load counterfactual pairs ----
    cf_pairs: List[Dict[str, Any]] = []
    if cf_path.exists():
        with open(cf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cf_pairs.append(json.loads(line))
        print(f"[eval] CF pairs loaded: {len(cf_pairs)}")
    else:
        print(f"[eval] WARNING: CF pairs file not found at {cf_path}. Skipping CF evaluation.")

    # ---- load pre-computed CSV metrics (optional) ----
    csv_metrics: Dict[str, Dict[str, float]] = {}
    if MODEL_COMPARISON_CSV:
        csv_path = _resolve(MODEL_COMPARISON_CSV)
        if csv_path.exists():
            csv_metrics = _load_csv_metrics(csv_path)
            print(f"[eval] CSV metrics loaded for: {list(csv_metrics.keys())}")
        else:
            print(f"[eval] WARNING: MODEL_COMPARISON_CSV not found: {csv_path}")

    # ---- evaluate each model ----
    all_metrics: Dict[str, Any] = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n{'='*70}")
        print(f"[eval] Evaluating model: {model_name}")
        print(f"{'='*70}")

        model, tokenizer, label2id, id2label = load_classifier(str(model_dir), device=DEVICE)

        # ---- Sanity: print label mappings ----
        print(f"[eval] label2id = {label2id}")
        print(f"[eval] id2label = {id2label}")

        # ==================================================================
        # 1) Standard evaluation on original test
        # ==================================================================
        print("[eval] 1) Standard evaluation on original test set...")
        orig_labels = [label2id[row["label"]] for _, row in df_test.iterrows()]

        # --- Try loading cached per-sample predictions (test_predictions.csv) ---
        cached = _load_cached_preds(model_dir, df_test, label2id, id2label)

        if cached is not None:
            orig_preds, orig_probs = cached
            std_metrics = compute_standard_metrics(orig_labels, orig_preds.tolist(), orig_probs, id2label)
            print(f"  Accuracy: {std_metrics['accuracy']:.4f}  Macro-F1: {std_metrics['f1_macro']:.4f}  (from cache)")
        else:
            # Run inference on test set
            orig_texts = [
                _make_input(row["context"], row["prompt"], row["response"])
                for _, row in df_test.iterrows()
            ]
            orig_preds, orig_probs = predict_batch(model, tokenizer, orig_texts, BATCH_SIZE, desc="Original test")
            std_metrics = compute_standard_metrics(orig_labels, orig_preds.tolist(), orig_probs, id2label)
            print(f"  Accuracy: {std_metrics['accuracy']:.4f}  Macro-F1: {std_metrics['f1_macro']:.4f}")

        # --- Override standard metrics from CSV if available ---
        if csv_metrics and model_name in csv_metrics:
            csv_m = csv_metrics[model_name]
            std_metrics["accuracy"] = csv_m["accuracy"]
            std_metrics["f1_macro"] = csv_m["f1_macro"]
            std_metrics["f1_weighted"] = csv_m["f1_weighted"]
            print(f"  [csv] Standard metrics overridden from {MODEL_COMPARISON_CSV}")

        # ==================================================================
        # 2) Counterfactual evaluation
        # ==================================================================
        cf_metrics_all: Dict[str, Any] = {}
        cf_bucket_metrics: Dict[str, Any] = {}

        if cf_pairs:
            print("[eval] 2) Counterfactual evaluation...")

            # Build lookup: id -> original probs/pred
            orig_lookup: Dict[str, Dict] = {}
            for i, row in df_test.iterrows():
                orig_lookup[row["id"]] = {
                    "pred": int(orig_preds[i]),
                    "probs": orig_probs[i].tolist(),
                }

            # Predict on counterfactuals
            cf_texts: List[str] = []
            cf_meta: List[Dict] = []
            for cp in cf_pairs:
                cf_resp = cp.get("counterfactual_response", "")
                if not cf_resp:
                    cf_resp = cp.get("response", "")  # fallback
                cf_texts.append(
                    _make_input(cp["context"], cp["prompt"], cf_resp)
                )
                cf_meta.append(cp)

            cf_preds, cf_probs = predict_batch(model, tokenizer, cf_texts, BATCH_SIZE, desc="Counterfactual")

            # Build records
            cf_eval_records: List[Dict[str, Any]] = []
            for i, cp in enumerate(cf_meta):
                orig_info = orig_lookup.get(cp["id"], {"pred": -1, "probs": [0] * len(label2id)})
                edit_stats = cp.get("edit_stats", {})
                edit_size = get_edit_size(edit_stats)
                rec = {
                    "id": cp["id"],
                    "bucket": cp["bucket"],
                    "y_true": cp["y_true"],
                    "y_cf_expected": cp["y_cf_expected"],
                    "pred_orig": orig_info["pred"],
                    "pred_cf": int(cf_preds[i]),
                    "probs_orig": orig_info["probs"],
                    "probs_cf": cf_probs[i].tolist(),
                    "edit_size": edit_size,
                    "valid_cf": cp.get("valid_cf", False),
                }
                cf_eval_records.append(rec)

            cf_metrics_all = compute_cf_metrics(cf_eval_records, label2id)
            cf_bucket_metrics = compute_bucket_metrics(cf_eval_records, label2id)

            print(f"  CFR: {cf_metrics_all.get('CFR', 0):.4f}  DCS: {cf_metrics_all.get('DCS', 0):.4f}")
            for bk, bm in cf_bucket_metrics.items():
                print(f"    Bucket {bk}: CFR={bm.get('CFR',0):.4f} DCS={bm.get('DCS',0):.4f} n={bm.get('n_valid',0)}")

            # ---- Sanity: valid_cf distribution ----
            valid_by_bucket: Dict[str, int] = defaultdict(int)
            total_valid = 0
            for rec in cf_eval_records:
                if rec["valid_cf"]:
                    valid_by_bucket[rec["bucket"]] += 1
                    total_valid += 1
            print(f"  [sanity] N_valid_cf = {total_valid}  bucket dist = {dict(sorted(valid_by_bucket.items()))}")

            # ---- Sanity: edit_size distribution ----
            es = cf_metrics_all.get("edit_size_stats", {})
            if es:
                print(f"  [sanity] edit_size: min={es.get('min','?')}  median={es.get('median','?')}  p90={es.get('p90','?')}  max={es.get('max','?')}  missing={es.get('missing_count',0)}")
            mesr_n = cf_metrics_all.get("mesr_subset_counts", {})
            for k in [1, 3, 5, 10]:
                print(f"  [sanity] n(edit_size<={k:>2d}) = {mesr_n.get(f'n_le_{k}', '?')}")

            # ---- Sanity: MESR@k ----
            for k_tag in ["MESR@1", "MESR@3", "MESR@5", "MESR@10"]:
                val = cf_metrics_all.get(k_tag)
                print(f"  [sanity] {k_tag} = {val if val is None else f'{val:.4f}'}")

            # Write preds_counterfactual.jsonl
            preds_cf_path = out_dir / f"preds_counterfactual_{model_name}.jsonl"
            with open(preds_cf_path, "w", encoding="utf-8") as f:
                for rec in cf_eval_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # ==================================================================
        # 3) Stability (non-semantic edits)
        # ==================================================================
        stability_metrics: Dict[str, Any] = {}
        if DO_STABILITY:
            print("[eval] 3) Stability evaluation (non-semantic edits)...")
            ns_texts: List[str] = []
            ns_orig_preds_list: List[int] = []
            ns_indices: List[int] = []
            for i, row in df_test.iterrows():
                ns = generate_nonsemantic_edits(row["context"], row["prompt"], row["response"])
                if ns is not None:
                    ns_texts.append(_make_input(row["context"], row["prompt"], ns))
                    ns_orig_preds_list.append(int(orig_preds[i]))
                    ns_indices.append(i)

            # Subsample if STABILITY_N > 0
            if STABILITY_N > 0 and len(ns_texts) > STABILITY_N:
                idxs = list(range(len(ns_texts)))
                random.shuffle(idxs)
                idxs = idxs[:STABILITY_N]
                ns_texts = [ns_texts[j] for j in idxs]
                ns_orig_preds_list = [ns_orig_preds_list[j] for j in idxs]
                ns_indices = [ns_indices[j] for j in idxs]

            if ns_texts:
                ns_preds, ns_probs = predict_batch(model, tokenizer, ns_texts, BATCH_SIZE, desc="Stability")
                flips = sum(1 for op, np_ in zip(ns_orig_preds_list, ns_preds) if op != np_)
                flip_rate = flips / len(ns_texts)
                stability_score = 1.0 - flip_rate

                # SG = CFR_semantic - flip_rate_nonsemantic
                cfr_val = cf_metrics_all.get("CFR", 0)
                sensitivity_gap = cfr_val - flip_rate

                stability_metrics = {
                    "n_samples": len(ns_texts),
                    "flip_rate_nonsemantic": float(flip_rate),
                    "stability_score": float(stability_score),
                    "sensitivity_gap": float(sensitivity_gap),
                    "sg_definition": "CFR_semantic - flip_rate_nonsemantic",
                }
                print(f"  Stability Score: {stability_score:.4f}  Flip Rate: {flip_rate:.4f}  SG: {sensitivity_gap:.4f}")
            else:
                print("  No non-semantic edits generated.")

        # ==================================================================
        # 4) No-context ablation
        # ==================================================================
        ablation_metrics: Dict[str, Any] = {}
        if DO_NO_CONTEXT and cf_pairs:
            print("[eval] 4) No-context ablation...")

            # Re-predict counterfactuals with empty context
            abl_texts: List[str] = []
            for cp in cf_pairs:
                cf_resp = cp.get("counterfactual_response", "")
                if not cf_resp:
                    cf_resp = cp.get("response", "")
                abl_texts.append(_make_input("", cp["prompt"], cf_resp))

            abl_preds, abl_probs = predict_batch(model, tokenizer, abl_texts, BATCH_SIZE, desc="Ablation CF")

            # Also predict original with empty context
            abl_orig_texts = [
                _make_input("", row["prompt"], row["response"])
                for _, row in df_test.iterrows()
            ]
            abl_orig_preds, abl_orig_probs = predict_batch(model, tokenizer, abl_orig_texts, BATCH_SIZE, desc="Ablation orig")

            # Standard metrics on no-context original
            abl_std = compute_standard_metrics(orig_labels, abl_orig_preds.tolist(), abl_orig_probs, id2label)

            # Build CF records for ablation
            orig_lookup_abl: Dict[str, Dict] = {}
            for i, row in df_test.iterrows():
                orig_lookup_abl[row["id"]] = {
                    "pred": int(abl_orig_preds[i]),
                    "probs": abl_orig_probs[i].tolist(),
                }

            abl_cf_records: List[Dict[str, Any]] = []
            for i, cp in enumerate(cf_pairs):
                orig_info = orig_lookup_abl.get(cp["id"], {"pred": -1, "probs": [0] * len(label2id)})
                edit_stats = cp.get("edit_stats", {})
                edit_size = get_edit_size(edit_stats)
                rec = {
                    "id": cp["id"],
                    "bucket": cp["bucket"],
                    "y_true": cp["y_true"],
                    "y_cf_expected": cp["y_cf_expected"],
                    "pred_orig": orig_info["pred"],
                    "pred_cf": int(abl_preds[i]),
                    "probs_orig": orig_info["probs"],
                    "probs_cf": abl_probs[i].tolist(),
                    "edit_size": edit_size,
                    "valid_cf": cp.get("valid_cf", False),
                }
                abl_cf_records.append(rec)

            abl_cf_metrics = compute_cf_metrics(abl_cf_records, label2id)
            abl_bucket_metrics = compute_bucket_metrics(abl_cf_records, label2id)

            # CFR drop per bucket
            cfr_drops: Dict[str, float] = {}
            for bk in ["A", "B", "C", "D"]:
                full_cfr = cf_bucket_metrics.get(bk, {}).get("CFR", 0)
                abl_cfr = abl_bucket_metrics.get(bk, {}).get("CFR", 0)
                cfr_drops[bk] = full_cfr - abl_cfr

            ablation_metrics = {
                "standard_no_context": abl_std,
                "cf_no_context": abl_cf_metrics,
                "cf_no_context_buckets": abl_bucket_metrics,
                "cfr_drop_per_bucket": cfr_drops,
            }
            print(f"  No-context Acc: {abl_std['accuracy']:.4f}  CFR: {abl_cf_metrics.get('CFR',0):.4f}")
            print(f"  CFR drops by bucket: {cfr_drops}")

        # ==================================================================
        # Assemble metrics
        # ==================================================================
        model_metrics: Dict[str, Any] = {
            "model_name": model_name,
            "model_id": json.load(open(model_dir / "training_config.json", encoding="utf-8"))["model_id"],
            "standard": std_metrics,
        }
        if cf_metrics_all:
            model_metrics["counterfactual"] = cf_metrics_all
            model_metrics["counterfactual_buckets"] = cf_bucket_metrics
        if stability_metrics:
            model_metrics["stability"] = stability_metrics
        if ablation_metrics:
            model_metrics["ablation"] = ablation_metrics

        all_metrics[model_name] = model_metrics

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # ---- write metrics_summary.json ----
    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Metrics summary written to {summary_path}")
    print("[run_eval] Patched: robust edit_size + monotonic MESR sanity + SG fixed.")


if __name__ == "__main__":
    main()
