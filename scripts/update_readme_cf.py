#!/usr/bin/env python3
"""
Append Counterfactual Dataset Validation + Counterfactual Evaluation Insights
to README.md without removing existing content.

Reads:
  - exp_out/metrics_summary.json
  - exp_out/validation_report.json
  - exp_out/counterfactual_pairs.jsonl  (for edit_size / sentence_change stats)
"""

import json, pathlib, statistics, collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP  = ROOT / "exp_out"

# ── helpers ──────────────────────────────────────────────────────────────────

def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

def pct(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v*100:.{decimals}f}%"

def pct_raw(v, decimals=2):
    """v is already in %."""
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}%"

# ── load data ────────────────────────────────────────────────────────────────

with open(EXP / "metrics_summary.json", encoding="utf-8") as f:
    metrics = json.load(f)

with open(EXP / "validation_report.json", encoding="utf-8") as f:
    vr = json.load(f)

# load counterfactual_pairs for edit_size / sentence_change stats
pairs = []
with open(EXP / "counterfactual_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))

# ── Part A: Counterfactual Dataset Validation ─────────────────────────────

total = vr.get("total", "N/A")
total_valid = vr.get("total_valid", "N/A")
coverage_pct = vr.get("coverage_pct", "N/A")

bucket_cov = vr.get("bucket_coverage", {})
BUCKETS = ["A", "B", "C", "D"]

# edit_size stats from valid pairs
valid_pairs = [p for p in pairs if p.get("valid_cf")]
edit_sizes = []
sentence_changes = []
for p in valid_pairs:
    es = p.get("edit_stats", {})
    tok = es.get("token_edits")
    if tok is not None:
        edit_sizes.append(tok)
    sc = es.get("sentence_change")
    if sc is not None:
        sentence_changes.append(sc)

def safe_stat(lst, fn, label="N/A"):
    try:
        return fn(lst)
    except Exception:
        return label

edit_min = safe_stat(edit_sizes, min)
edit_max = safe_stat(edit_sizes, max)
edit_median = safe_stat(edit_sizes, statistics.median)
edit_mean = safe_stat(edit_sizes, statistics.mean)
edit_p90 = safe_stat(edit_sizes, lambda l: sorted(l)[int(len(l)*0.9)])

# sentence_change distribution
sc_counter = collections.Counter(sentence_changes)

# edit_size bin counts  (<=1, <=3, <=5, <=10, >10)
def bin_count(lst, upper):
    return sum(1 for x in lst if x <= upper)

es_le1 = bin_count(edit_sizes, 1)
es_le3 = bin_count(edit_sizes, 3)
es_le5 = bin_count(edit_sizes, 5)
es_le10 = bin_count(edit_sizes, 10)
es_gt10 = len(edit_sizes) - es_le10

# top invalid reasons  (aggregate across buckets, top 10)
inv_hist = vr.get("invalid_reason_histogram", {})
agg_reasons = collections.Counter()
for bk, reasons in inv_hist.items():
    for reason, cnt in reasons.items():
        agg_reasons[reason] += cnt
top_invalid = agg_reasons.most_common(10)

# ── Build Part A markdown ────────────────────────────────────────────────

lines_a = []
lines_a.append("")
lines_a.append("## Counterfactual Dataset Validation")
lines_a.append("")
lines_a.append("### Overview")
lines_a.append("")
lines_a.append(f"| Metric | Value |")
lines_a.append(f"|--------|-------|")
lines_a.append(f"| Total CF pairs generated | {total} |")
lines_a.append(f"| Valid pairs | {total_valid} |")
lines_a.append(f"| Valid rate | {pct_raw(coverage_pct)}  |")
lines_a.append("")

# bucket distribution (total + valid)
lines_a.append("### Bucket Distribution")
lines_a.append("")
lines_a.append("| Bucket | Total | Valid | Coverage |")
lines_a.append("|--------|-------|-------|----------|")
for b in BUCKETS:
    bc = bucket_cov.get(b, {})
    lines_a.append(f"| {b} | {bc.get('total','N/A')} | {bc.get('valid','N/A')} | {pct_raw(bc.get('coverage_pct'))} |")
lines_a.append("")
lines_a.append("> Bucket A = intra-sentence factual fix; B = sentence deletion; C = sentence append; D = word-level replacement.")
lines_a.append("")

# top invalid reasons
lines_a.append("### Top Invalid Reasons (aggregated)")
lines_a.append("")
lines_a.append("| # | Reason | Count |")
lines_a.append("|---|--------|-------|")
for i, (reason, cnt) in enumerate(top_invalid, 1):
    lines_a.append(f"| {i} | `{reason}` | {cnt} |")
lines_a.append("")

# edit_size stats
lines_a.append("### Edit-Size Statistics (valid pairs only)")
lines_a.append("")
lines_a.append(f"| Stat | Value |")
lines_a.append(f"|------|-------|")
lines_a.append(f"| Count | {len(edit_sizes)} |")
lines_a.append(f"| Min | {edit_min} |")
lines_a.append(f"| Median | {fmt(edit_median, 1)} |")
lines_a.append(f"| Mean | {fmt(edit_mean, 2)} |")
lines_a.append(f"| P90 | {edit_p90} |")
lines_a.append(f"| Max | {edit_max} |")
lines_a.append("")

lines_a.append("**Edit-size bin counts:**")
lines_a.append("")
lines_a.append("| Bin | Count | % of valid |")
lines_a.append("|-----|-------|------------|")
n_valid_total = len(edit_sizes) if edit_sizes else 1
for label, cnt in [("≤1", es_le1), ("≤3", es_le3), ("≤5", es_le5), ("≤10", es_le10), (">10", es_gt10)]:
    lines_a.append(f"| {label} | {cnt} | {cnt/n_valid_total*100:.1f}% |")
lines_a.append("")

# sentence_change stats
lines_a.append("### Sentence-Change Distribution (valid pairs)")
lines_a.append("")
lines_a.append("| sentence_change | Count | % |")
lines_a.append("|-----------------|-------|---|")
for sc_val in sorted(sc_counter.keys()):
    cnt = sc_counter[sc_val]
    lines_a.append(f"| {sc_val} | {cnt} | {cnt/n_valid_total*100:.1f}% |")
lines_a.append("")

# limitations
lines_a.append("### Limitations & Action Items")
lines_a.append("")
lines_a.append("- **Bucket B has lowest coverage (63.3%)**: primary failure is `bucket_B_sentence_change_not_minus1` — the generator often fails to produce a proper sentence-deletion CF.")
lines_a.append("- **Bucket D**: `D_target_not_in_response` (96) and `bucket_D_no_contradiction` (84) dominate invalid cases — the small entity replacement sometimes cannot find the target span.")
lines_a.append("- **Minimality failures** in Buckets A and C can be reduced by tightening the token-edit budget or adding a post-hoc minimality filter.")
lines_a.append("- Consider re-running failed Bucket-B samples with a specialized deletion prompt.")
lines_a.append("")

# ── Part B: Counterfactual Evaluation Insights ───────────────────────────

MODEL_ORDER = ["Qwen_Qwen3_1.7B", "Qwen_Qwen3_4B", "Qwen_Qwen3_8B"]
DISPLAY_NAMES = {
    "Qwen_Qwen3_1.7B": "Qwen3-1.7B",
    "Qwen_Qwen3_4B":   "Qwen3-4B",
    "Qwen_Qwen3_8B":   "Qwen3-8B",
}

lines_b = []
lines_b.append("## Counterfactual Evaluation Insights")
lines_b.append("")

# B1 – Setup recap
lines_b.append("### B1 — Setup Recap")
lines_b.append("")
lines_b.append(f"- **Test set size**: 2,000 samples (3 classes: no / intrinsic / extrinsic)")
lines_b.append(f"- **Counterfactual valid pairs**: {total_valid} / {total} ({pct_raw(coverage_pct)} coverage)")
bucket_dist_str = ", ".join(
    f"{b}={bucket_cov.get(b,{}).get('valid','?')}" for b in BUCKETS
)
lines_b.append(f"- **Bucket distribution (valid)**: {bucket_dist_str}")
lines_b.append(f"- **Metrics**: Accuracy, Macro-F1 (standard); CFR (counterfactual flip rate = fraction of valid CFs where model correctly flips prediction), DCS (decision change score), MESR@k (minimal-edit success rate on subsets with edit_size ≤ k), Stability Score = 1 − flip_rate_nonsemantic, Sensitivity Gap = CFR − flip_rate_nonsemantic.")
lines_b.append("")

# B2 – Key findings
lines_b.append("### B2 — Key Findings")
lines_b.append("")

# prepare per-model summaries for bullets
bullets = []
# 1) Standard perf scaling
accs = {m: metrics[m]["standard"]["accuracy"] for m in MODEL_ORDER}
bullets.append(f"**Standard accuracy scales with model size**: Qwen3-1.7B {fmt(accs['Qwen_Qwen3_1.7B'])} → Qwen3-4B {fmt(accs['Qwen_Qwen3_4B'])} → Qwen3-8B {fmt(accs['Qwen_Qwen3_8B'])}.")

# 2) CFR does NOT scale
cfrs = {m: metrics[m]["counterfactual"]["CFR"] for m in MODEL_ORDER}
bullets.append(f"**CFR does NOT scale with model size**: Qwen3-1.7B CFR={fmt(cfrs['Qwen_Qwen3_1.7B'])} ≈ Qwen3-4B CFR={fmt(cfrs['Qwen_Qwen3_4B'])} > Qwen3-8B CFR={fmt(cfrs['Qwen_Qwen3_8B'])}. The largest model is *less* sensitive to minimal semantic edits.")

# 3) Bucket B is hardest
b_cfrs = {m: metrics[m]["counterfactual_buckets"]["B"]["CFR"] for m in MODEL_ORDER}
bullets.append(f"**Bucket B (sentence deletion) is the hardest**: CFR_B ranges from {fmt(min(b_cfrs.values()))} to {fmt(max(b_cfrs.values()))}, far below the overall CFR. Deletion removes stylistic cues models rely on.")

# 4) Bucket C easiest for smaller models
c_cfrs = {m: metrics[m]["counterfactual_buckets"]["C"]["CFR"] for m in MODEL_ORDER}
bullets.append(f"**Bucket C (sentence append) is easiest for smaller models**: CFR_C = {fmt(c_cfrs['Qwen_Qwen3_1.7B'])} (1.7B), {fmt(c_cfrs['Qwen_Qwen3_4B'])} (4B), but drops to {fmt(c_cfrs['Qwen_Qwen3_8B'])} (8B) — the 8B model resists appended sentences.")

# 5) Bucket D
d_cfrs = {m: metrics[m]["counterfactual_buckets"]["D"]["CFR"] for m in MODEL_ORDER}
bullets.append(f"**Bucket D (word replacement) shows inverse scaling for 8B**: CFR_D = {fmt(d_cfrs['Qwen_Qwen3_8B'])} (8B) > {fmt(d_cfrs['Qwen_Qwen3_1.7B'])} (1.7B), suggesting the largest model is more sensitive to entity-level swaps.")

# 6) Stability
stab_scores = {m: metrics[m]["stability"]["stability_score"] for m in MODEL_ORDER}
flip_rates = {m: metrics[m]["stability"]["flip_rate_nonsemantic"] for m in MODEL_ORDER}
bullets.append(f"**Stability decreases with scale**: SS = {fmt(stab_scores['Qwen_Qwen3_1.7B'])} (1.7B) → {fmt(stab_scores['Qwen_Qwen3_4B'])} (4B) → {fmt(stab_scores['Qwen_Qwen3_8B'])} (8B). Larger models are more volatile on non-semantic perturbations.")

# 7) Sensitivity gap
sgs = {m: metrics[m]["stability"]["sensitivity_gap"] for m in MODEL_ORDER}
bullets.append(f"**Sensitivity Gap (SG = CFR − flip_rate)**: Qwen3-1.7B SG={fmt(sgs['Qwen_Qwen3_1.7B'])} (good), Qwen3-4B SG={fmt(sgs['Qwen_Qwen3_4B'])}, Qwen3-8B SG={fmt(sgs['Qwen_Qwen3_8B'])} (≈0, meaning semantic and noise flips are equally likely).")

# 8) No-context ablation
no_ctx_accs = {m: metrics[m]["ablation"]["standard_no_context"]["accuracy"] for m in MODEL_ORDER}
no_ctx_cfrs = {m: metrics[m]["ablation"]["cf_no_context"]["CFR"] for m in MODEL_ORDER}
bullets.append(f"**No-context accuracy and CFR**: without context, accuracy drops to {fmt(no_ctx_accs['Qwen_Qwen3_1.7B'])}–{fmt(no_ctx_accs['Qwen_Qwen3_8B'])}. No-context CFR for 1.7B ({fmt(no_ctx_cfrs['Qwen_Qwen3_1.7B'])}) < with-context CFR ({fmt(cfrs['Qwen_Qwen3_1.7B'])}), confirming models use context to detect hallucinations.")

# 9) ΔConf
delta_confs = {m: metrics[m]["counterfactual"]["delta_conf_mean"] for m in MODEL_ORDER}
bullets.append(f"**Confidence drops on CFs**: mean ΔConf = {fmt(delta_confs['Qwen_Qwen3_1.7B'])} (1.7B), {fmt(delta_confs['Qwen_Qwen3_4B'])} (4B), {fmt(delta_confs['Qwen_Qwen3_8B'])} (8B). Larger models show steeper confidence drops, indicating they detect semantic changes but still predict the wrong label.")

for b in bullets:
    lines_b.append(f"- {b}")
lines_b.append("")

# ── B3 Tables ────────────────────────────────────────────────────────────

# TABLE 1 — Standard Performance
lines_b.append("### B3 — Evaluation Tables")
lines_b.append("")
lines_b.append("#### Table 1 — Standard Performance")
lines_b.append("")
lines_b.append("| Model | Accuracy | Macro-F1 | Weighted-F1 |")
lines_b.append("|-------|----------|----------|-------------|")
for m in MODEL_ORDER:
    s = metrics[m]["standard"]
    lines_b.append(f"| {DISPLAY_NAMES[m]} | {fmt(s['accuracy'])} | {fmt(s['f1_macro'])} | {fmt(s['f1_weighted'])} |")
lines_b.append("")
lines_b.append("> *Reading guide*: Standard 3-class hallucination detection on the 2 000-sample test set. Higher is better.")
lines_b.append("")

# TABLE 2 — Counterfactual Overall
lines_b.append("#### Table 2 — Counterfactual Overall")
lines_b.append("")
lines_b.append("| Model | CFR | DCS | n_valid | coverage | ΔConf_mean | ΔConf_median |")
lines_b.append("|-------|-----|-----|---------|----------|------------|--------------|")
for m in MODEL_ORDER:
    cf = metrics[m]["counterfactual"]
    lines_b.append(f"| {DISPLAY_NAMES[m]} | {fmt(cf['CFR'])} | {fmt(cf['DCS'])} | {cf['n_valid']} | {pct(cf['coverage_rate'])} | {fmt(cf['delta_conf_mean'])} | {fmt(cf['delta_conf_median'])} |")
lines_b.append("")
lines_b.append("> *Reading guide*: CFR = fraction of valid CFs where the model correctly flips its prediction to the expected label. DCS = fraction where the model changes its decision at all (even if to a wrong class). ΔConf = change in predicted-class confidence (negative = model becomes less confident).")
lines_b.append("")

# TABLE 3 — Bucket-wise CFR
lines_b.append("#### Table 3 — Counterfactual Bucket-wise CFR")
lines_b.append("")
lines_b.append("| Model | A_CFR | B_CFR | C_CFR | D_CFR |")
lines_b.append("|-------|-------|-------|-------|-------|")
for m in MODEL_ORDER:
    bk = metrics[m]["counterfactual_buckets"]
    line = f"| {DISPLAY_NAMES[m]}"
    for b in BUCKETS:
        line += f" | {fmt(bk[b]['CFR'])}"
    line += " |"
    lines_b.append(line)
# n_valid row
nv_line = "| *n_valid*"
for b in BUCKETS:
    nv_line += f" | {bucket_cov.get(b,{}).get('valid','N/A')}"
nv_line += " |"
lines_b.append(nv_line)
lines_b.append("")
lines_b.append("> *Reading guide*: A = intra-sentence factual fix, B = sentence deletion, C = sentence append, D = word-level entity replacement. Bucket B is consistently the hardest across all models.")
lines_b.append("")

# TABLE 4 — Robustness
lines_b.append("#### Table 4 — Robustness (Stability + No-Context)")
lines_b.append("")
lines_b.append("| Model | StabilityScore | FlipRate_nonsem | SG | NoCtxAcc | NoCtxCFR |")
lines_b.append("|-------|----------------|-----------------|-----|----------|----------|")
for m in MODEL_ORDER:
    st = metrics[m]["stability"]
    abl = metrics[m]["ablation"]
    lines_b.append(
        f"| {DISPLAY_NAMES[m]} "
        f"| {fmt(st['stability_score'])} "
        f"| {fmt(st['flip_rate_nonsemantic'])} "
        f"| {fmt(st['sensitivity_gap'])} "
        f"| {fmt(abl['standard_no_context']['accuracy'])} "
        f"| {fmt(abl['cf_no_context']['CFR'])} |"
    )
lines_b.append("")
lines_b.append("> *Reading guide*: SS = 1 − flip_rate on non-semantic perturbations (higher = more stable). SG = CFR − flip_rate (positive = model is more sensitive to *semantic* changes than to noise, which is desirable). NoCtxAcc/NoCtxCFR = performance when context is removed — measures reliance on superficial cues.")
lines_b.append("")

# TABLE 5 — MESR@k
lines_b.append("#### Table 5 — MESR@k & Subset Counts")
lines_b.append("")
lines_b.append("| Model | n≤1 | MESR@1 | n≤3 | MESR@3 | n≤5 | MESR@5 | n≤10 | MESR@10 |")
lines_b.append("|-------|-----|--------|-----|--------|-----|--------|------|---------|")
for m in MODEL_ORDER:
    cf = metrics[m]["counterfactual"]
    sc = cf.get("mesr_subset_counts", {})
    lines_b.append(
        f"| {DISPLAY_NAMES[m]} "
        f"| {sc.get('n_le_1','N/A')} | {fmt(cf.get('MESR@1'))} "
        f"| {sc.get('n_le_3','N/A')} | {fmt(cf.get('MESR@3'))} "
        f"| {sc.get('n_le_5','N/A')} | {fmt(cf.get('MESR@5'))} "
        f"| {sc.get('n_le_10','N/A')} | {fmt(cf.get('MESR@10'))} |"
    )
lines_b.append("")
lines_b.append("> *Reading guide*: MESR@k = counterfactual flip rate computed only on the subset of pairs with `edit_size ≤ k`. Because each threshold defines a *different* subset, MESR@k may decrease as k increases (the added samples are harder, not easier). n≤k = subset size.")
lines_b.append("")

# ── B4 Interpretation & Next Steps ───────────────────────────────────────

lines_b.append("### B4 — Interpretation & Next Steps")
lines_b.append("")
lines_b.append(
    "**Why is Bucket B so difficult?** "
    "Bucket B counterfactuals delete an entire sentence that originally introduced an extrinsic hallucination. "
    "When an extrinsic sentence is removed, the remaining response becomes *shorter* and merely faithful — "
    "but models trained on length and stylistic cues often still predict \"extrinsic\" because "
    "the prompt style or remaining content pattern hasn't changed. "
    "The consistently low CFR_B (0.10–0.15) across all scales confirms that "
    "current detectors depend heavily on surface signals beyond the actual semantic content."
)
lines_b.append("")
lines_b.append(
    "**Model-scale trade-off: standard ↑ but CFR ↓.** "
    "While standard accuracy rises from 0.7960 (1.7B) to 0.8455 (8B), "
    "overall CFR *decreases* from 0.5338 to 0.3647. "
    "The larger model is better at memorizing training-distribution patterns "
    "but worse at adapting to minimal semantic perturbations. "
    "This reveals an important gap: "
    "high test-set accuracy alone does not guarantee robust, causally grounded detection."
)
lines_b.append("")
lines_b.append(
    "**Stability vs. sensitivity.** "
    "Qwen3-8B's flip_rate_nonsemantic is 0.375 (vs. 0.275 for 1.7B), "
    "meaning it flips predictions on 37.5% of *non-semantic* perturbations. "
    "Its near-zero Sensitivity Gap (SG ≈ −0.01) indicates it cannot distinguish "
    "between meaningful semantic edits and surface noise. "
    "A robust detector should have high CFR and low flip_rate — i.e., high SG."
)
lines_b.append("")
lines_b.append(
    "**Proposed next steps:** "
    "(1) Analyse confusion matrices per bucket to locate which class transitions "
    "the model struggles with (e.g., intrinsic→no vs. extrinsic→no). "
    "(2) Bin CFR by edit_size (≤3, 4–10, >10) to quantify whether minimal edits "
    "are harder or easier to detect. "
    "(3) Augment training with counterfactual pairs to improve causal sensitivity. "
    "(4) Investigate adversarial training or consistency regularisation to reduce "
    "non-semantic flip rate while preserving CFR."
)
lines_b.append("")

# ── Assemble & write ─────────────────────────────────────────────────────

readme_path = ROOT / "README.md"
existing = readme_path.read_text(encoding="utf-8")

# Remove previous generated sections if re-running
SECTION_MARKERS = [
    "## Counterfactual Dataset Validation",
    "## Counterfactual Evaluation Insights",
]
for marker in SECTION_MARKERS:
    idx = existing.find(marker)
    if idx != -1:
        existing = existing[:idx].rstrip()

new_content = existing + "\n" + "\n".join(lines_a) + "\n" + "\n".join(lines_b)
readme_path.write_text(new_content, encoding="utf-8")

# ── Console summary ─────────────────────────────────────────────────────

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total CF pairs: {total}  |  Valid: {total_valid}  |  Rate: {pct_raw(coverage_pct)}")
print()
print("Top invalid reasons:")
for reason, cnt in top_invalid[:5]:
    print(f"  {reason}: {cnt}")
print()
print("CFR per model:")
for m in MODEL_ORDER:
    print(f"  {DISPLAY_NAMES[m]:12s}  CFR = {fmt(metrics[m]['counterfactual']['CFR'])}")
print()
print("README updated with validation summary + multi-table evaluation results.")
