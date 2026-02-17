#!/usr/bin/env python3
"""
build_counterfactuals.py
========================
Generate counterfactual pairs from the test set using a HuggingFace causal LM.
Supports 4 buckets:
  A) intrinsic  -> no       (repair contradiction)
  B) extrinsic  -> no       (delete unsupported span)
  C) no         -> extrinsic (inject unsupported claim)
  D) no         -> intrinsic (inject contradiction)

Deterministic safety: validation uses heuristic flags computed from
(context, response, counterfactual_response) — no dependency on LLM
self-report.  All safety_check fields in the output JSONL are assigned
by the validator, not by the generator.

Resume-safe: skips IDs already present in the output JSONL.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET_MAP = {
    "A": {"src": "intrinsic", "tgt": "no"},
    "B": {"src": "extrinsic", "tgt": "no"},
    "C": {"src": "no", "tgt": "extrinsic"},
    "D": {"src": "no", "tgt": "intrinsic"},
}

BUCKET_TEMP = {"A": 0.2, "B": 0.2, "C": 0.4, "D": 0.2}

# Default validation thresholds
MAX_TOKEN_EDITS = 15
MAX_SENTENCE_CHANGE = 1

# Per-bucket max_token_edits overrides
MAX_TOKEN_EDITS_A = 25
MAX_TOKEN_EDITS_B = 15
MAX_TOKEN_EDITS_C = 15
MAX_TOKEN_EDITS_D = 10

BUCKET_MAX_TOKEN_EDITS = {
    "A": MAX_TOKEN_EDITS_A,
    "B": MAX_TOKEN_EDITS_B,
    "C": MAX_TOKEN_EDITS_C,
    "D": MAX_TOKEN_EDITS_D,
}

# Deterministic safety thresholds
NOVEL_OUTSIDE_CONTEXT_THRESHOLD = 6    # max novel tokens outside context (bucket A, tuned for VN)
OVERLAP_DROP_THRESHOLD = 0.15          # max allowed drop in context overlap (bucket A)
JACCARD_UNSUPPORTED_THRESHOLD = 0.8    # Jaccard threshold for unsupported claim (bucket C)
MIN_ADDED_SENTENCE_TOKENS = 5         # min tokens in appended sentence (bucket C)

# Bucket B deletion-based validation thresholds
MAX_DELETED_TOKENS_B = 80             # max tokens deleted in bucket B
MIN_DELETED_TOKENS_B = 3              # min tokens deleted in bucket B

# Clause splitting markers for Vietnamese (bucket B)
VN_CLAUSE_MARKERS = {"và", "đồng thời", "nhờ", "do", "ngoài ra", "hoặc", "hay", "cũng như"}

# Vietnamese negation markers for contradiction heuristic (bucket D)
VN_NEGATION_MARKERS = {"không", "chưa", "chẳng"}

# ---------------------------------------------------------------------------
# Prompts — simplified: no safety_check required from LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a counterfactual editor for a hallucination detection dataset.\n"
    "Rules:\n"
    "1) Output MUST be valid JSON only. No extra text.\n"
    "2) Make the MINIMAL edit to the given Response.\n"
    "3) Keep the original wording/style as much as possible.\n"
    "4) Follow the bucket-specific constraints exactly.\n"
    "5) If you cannot produce a valid counterfactual, output:\n"
    '   {"counterfactual_response": "", "rationale_short": "cannot", "edit_summary": []}\n'
    "\n"
    "Required output JSON schema:\n"
    "{\n"
    '  "counterfactual_response": "<edited text>",\n'
    '  "edit_summary": ["<change description>"],\n'
    '  "rationale_short": "<brief reasoning>"\n'
    "}"
)

SYSTEM_PROMPT_SELECTOR = (
    "You are a selector tool for a hallucination detection dataset.\n"
    "Rules:\n"
    "1) Output MUST be valid JSON only. No extra text.\n"
    "2) Follow the schema and constraints EXACTLY.\n"
    "3) NEVER rewrite the Response. Only select spans/positions.\n"
)

BUCKET_USER_TEMPLATES: Dict[str, str] = {}

BUCKET_USER_TEMPLATES["A"] = (
    "Nhiệm vụ: Sửa Response để KHÔNG còn mâu thuẫn với Context (chuyển INTRINSIC → NO).\n"
    "Định nghĩa:\n"
    "- INTRINSIC: Response mâu thuẫn với Context.\n"
    "- NO: Response được Context hỗ trợ hoàn toàn.\n"
    "Ràng buộc BẮT BUỘC:\n"
    "- CHỈ được thay 1 cụm tối đa 5 token. KHÔNG được viết lại câu.\n"
    "- CHỈ dùng thông tin có trong Context. KHÔNG thêm thông tin mới.\n"
    "- KHÔNG được thêm các từ/cụm như: 'tuy nhiên', 'nhưng', 'thực tế', 'do đó'.\n"
    "- Giữ nguyên cấu trúc, số câu, và cách diễn đạt gốc.\n"
    "- Nếu không thể sửa tối thiểu, trả counterfactual_response=\"\".\n"
    "Input:\n"
    "[CONTEXT]\n{context}\n"
    "[PROMPT]\n{prompt}\n"
    "[RESPONSE]\n{response}\n"
    "Xuất JSON duy nhất."
)

BUCKET_USER_TEMPLATES["B"] = (
    "Bạn là công cụ CHỌN VỊ TRÍ CẦN XÓA để loại bỏ thông tin KHÔNG được CONTEXT hỗ trợ.\n"
    "TUYỆT ĐỐI không được viết lại RESPONSE. Chỉ được chọn 1 đơn vị để xóa.\n"
    "\n"
    "Quy tắc:\n"
    "- Nếu RESPONSE có >=2 câu (tách theo dấu chấm/chấm hỏi/chấm than): chọn delete_unit=\"sentence\"\n"
    "- Nếu RESPONSE chỉ 1 câu: chọn delete_unit=\"clause\" (tách theo dấu phẩy/chấm phẩy/và/đồng thời/nhờ/do)\n"
    "- index là vị trí 0-based của câu/mệnh đề cần xóa\n"
    "- Nếu không tìm được phần cần xóa, trả index=-1\n"
    "\n"
    "Schema JSON:\n"
    '{{"delete_unit": "sentence" | "clause", "index": <int>, "reason_short": "<lý do>"}}\n'
    "\n"
    "[CONTEXT]\n{context}\n"
    "[RESPONSE]\n{response}\n"
    "Xuất JSON duy nhất."
)

BUCKET_USER_TEMPLATES["C"] = (
    "Nhiệm vụ: Thêm đúng 1 câu thông tin KHÔNG có trong Context vào cuối Response (chuyển NO → EXTRINSIC).\n"
    "Định nghĩa:\n"
    "- NO: Response được Context hỗ trợ hoàn toàn.\n"
    "- EXTRINSIC: Response chứa ít nhất 1 thông tin không có trong Context.\n"
    "Ràng buộc BẮT BUỘC:\n"
    "- Thêm đúng 1 câu vào CUỐI Response. KHÔNG sửa phần Response gốc.\n"
    "- Câu thêm KHÔNG được nằm trong hoặc suy ra từ Context.\n"
    "- Câu thêm KHÔNG được mâu thuẫn với Context (phải là EXTRINSIC, không phải INTRINSIC).\n"
    "- KHÔNG được thêm các từ/cụm như: 'tuy nhiên', 'nhưng', 'thực tế', 'do đó'.\n"
    "Gợi ý mẫu câu thêm (chọn 1):\n"
    "1) 'Ngoài ra, dân số khoảng 2 triệu người.'\n"
    "2) 'Nó được thành lập vào năm 1800.'\n"
    "3) 'Khu vực này nằm gần biển.'\n"
    "4) 'Diện tích khoảng 500 km².'\n"
    "5) 'Nơi đây nổi tiếng với một lễ hội thường niên.'\n"
    "Input:\n"
    "[CONTEXT]\n{context}\n"
    "[PROMPT]\n{prompt}\n"
    "[RESPONSE]\n{response}\n"
    "Xuất JSON duy nhất."
)

BUCKET_USER_TEMPLATES["D"] = (
    "Bạn là công cụ CHỌN SPAN để tạo MÂU THUẪN (intrinsic) với CONTEXT bằng cách thay đúng 1 span trong RESPONSE.\n"
    "TUYỆT ĐỐI không được viết lại RESPONSE.\n"
    "\n"
    "Quy tắc:\n"
    "- target_text phải xuất hiện NGUYÊN VĂN trong RESPONSE (1–5 tokens liên tục)\n"
    "- replacement phải KHÁC target_text và cùng loại (số/năm/tên riêng)\n"
    "- Ưu tiên thay đổi: số, năm, tên riêng, tên địa danh\n"
    "- Chỉ 1 replace duy nhất. Không thêm câu, không giải thích.\n"
    "- Nếu không tìm được span phù hợp, trả target_text=\"\"\n"
    "\n"
    "Schema JSON:\n"
    '{{"target_text": "<span nguyên văn trong RESPONSE>", "replacement": "<span mới>", "type": "number" | "year" | "entity"}}\n'
    "\n"
    "[CONTEXT]\n{context}\n"
    "[RESPONSE]\n{response}\n"
    "Xuất JSON duy nhất."
)

RETRY_SUFFIX = (
    "\nOutput valid JSON only. Make fewer edits. Prefer <=5 token replacement or delete one clause. "
    "Do not rewrite."
)

RETRY_SUFFIX_B = (
    "\nOutput valid JSON only. Choose exactly ONE sentence or clause to delete. "
    "Use 0-based index. Do NOT rewrite the response."
)

RETRY_SUFFIX_D = (
    "\nOutput valid JSON only. target_text MUST appear verbatim in RESPONSE. "
    "Choose a number, year, or entity name (1-5 tokens). Do NOT rewrite."
)

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

# Sentence-ending regex: supports . ? ! … newline ; (Vietnamese-friendly)
_SENT_RE = re.compile(r"(?<=[.?!。…;])\s+|\n+")


def _clean(text: str) -> str:
    """Lightweight normalization (NFC + collapse whitespace)."""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def _remove_accents(text: str) -> str:
    """Strip Vietnamese diacritics/accents via Unicode NFD decomposition."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _normalize_text(text: str, remove_accents: bool = True) -> str:
    """Full normalization: NFC, lowercase, strip, collapse spaces, optionally remove accents."""
    t = unicodedata.normalize("NFC", str(text))
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    if remove_accents:
        t = _remove_accents(t)
    return t


def _normalize_for_compare(text: str) -> str:
    """Normalize for equality / subsequence checks: lower, strip, collapse spaces."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokenize_simple(text: str) -> List[str]:
    """Whitespace tokenizer (good enough for edit-distance)."""
    return text.split()


def _sentence_split(text: str) -> List[str]:
    """Split by sentence-ending punctuation (Vietnamese-friendly)."""
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _is_token_subsequence(sub_tokens: List[str], full_tokens: List[str]) -> bool:
    """Check if sub_tokens is a subsequence of full_tokens (order-preserving, two-pointer)."""
    if not sub_tokens:
        return True
    j = 0  # pointer into sub_tokens
    for ft in full_tokens:
        if ft == sub_tokens[j]:
            j += 1
            if j == len(sub_tokens):
                return True
    return j == len(sub_tokens)


def _levenshtein_tokens(a: List[str], b: List[str]) -> int:
    """Token-level Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # Use two-row optimization
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _overlap_ratio(text: str, ref: str) -> float:
    """Token-level Jaccard overlap."""
    a = set(_tokenize_simple(text.lower()))
    b = set(_tokenize_simple(ref.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _fuzzy_in(needle: str, haystack: str) -> bool:
    """Case-insensitive, accent-stripped substring check."""
    h = _normalize_text(haystack, remove_accents=True)
    n = _normalize_text(needle, remove_accents=True)
    return n in h


# Punctuation characters to strip for robust subsequence matching
_PUNCT_STRIP_CHARS = set('.,;:!?\u2026"\'()[]{}\u00ab\u00bb\u201c\u201d\u2018\u2019\u2013\u2014')


def _strip_punct_tokens(tokens: List[str]) -> List[str]:
    """Strip leading/trailing punctuation from each token, drop empty results.
    Used for punctuation-robust subsequence checks in bucket B."""
    out = []
    for t in tokens:
        stripped = t.strip()
        # Strip leading punct
        while stripped and stripped[0] in _PUNCT_STRIP_CHARS:
            stripped = stripped[1:]
        # Strip trailing punct
        while stripped and stripped[-1] in _PUNCT_STRIP_CHARS:
            stripped = stripped[:-1]
        if stripped:
            out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# Vietnamese sentence / clause splitting (for bucket B deterministic delete)
# ---------------------------------------------------------------------------

# Sentence-ending pattern for splitting
_SENT_SPLIT_RE = re.compile(r'(?<=[.?!。…;])\s+|\n+')

# Clause split pattern: comma, semicolon, and Vietnamese conjunctions
_CLAUSE_SPLIT_RE = re.compile(
    r'\s*[,;]\s*|\s+(?:và|đồng thời|nhờ|do|ngoài ra|hoặc|hay|cũng như)\s+',
    re.IGNORECASE,
)


def split_sentences_vn(text: str) -> List[str]:
    """Split Vietnamese text into sentences by punctuation (.?!…;) and newlines."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_clauses_vn(text: str) -> List[str]:
    """Split a single Vietnamese sentence into clauses.
    Splits on: comma, semicolon, and markers: và, đồng thời, nhờ, do, ngoài ra, etc.
    Only keeps clauses with >= 3 tokens to avoid overly fragmented splits.
    """
    text = text.strip()
    if not text:
        return []
    parts = _CLAUSE_SPLIT_RE.split(text)
    clauses = [p.strip() for p in parts if p.strip()]
    # Filter: only keep clauses with >= 3 tokens
    filtered = [c for c in clauses if len(_tokenize_simple(c)) >= 3]
    # If filtering removed too much, return original as single clause
    if len(filtered) < 2:
        return [text]
    return filtered


# ---------------------------------------------------------------------------
# Deterministic counterfactual builders (Bucket B & D)
# ---------------------------------------------------------------------------

def build_cf_B_by_deletion(
    response: str,
    selector: dict,
) -> Tuple[str, Dict[str, Any], str]:
    """Build counterfactual for bucket B by deterministic deletion.

    Args:
        response: original response text
        selector: parsed selector JSON with delete_unit, index, reason_short

    Returns:
        (cf_response, selector_meta, fail_reason)
        cf_response="" and fail_reason set if build fails.
    """
    delete_unit = selector.get("delete_unit", "sentence")
    index = selector.get("index", -1)
    reason_short = selector.get("reason_short", "")
    sel_meta = {
        "delete_unit": delete_unit,
        "index": index,
        "reason_short": reason_short,
    }

    # Abort signal from LLM
    if index == -1 or reason_short == "cannot_find":
        return "", sel_meta, "selector_cannot_find"

    if delete_unit == "sentence":
        sentences = split_sentences_vn(response)
        if not sentences:
            return "", sel_meta, "B_no_sentences_found"
        if index < 0 or index >= len(sentences):
            return "", sel_meta, f"B_sentence_index_oob({index}/{len(sentences)})"
        remaining = [s for i, s in enumerate(sentences) if i != index]
        if not remaining:
            return "", sel_meta, "B_would_delete_everything"
        # Ensure each sentence ends with punctuation before joining
        joined_parts = []
        for s in remaining:
            s = s.strip()
            if s and s[-1] not in ".?!;\u2026":
                s += "."
            joined_parts.append(s)
        cf_response = " ".join(joined_parts)

    elif delete_unit == "clause":
        clauses = split_clauses_vn(response)
        if len(clauses) < 2:
            return "", sel_meta, "B_single_clause_cannot_delete"
        if index < 0 or index >= len(clauses):
            return "", sel_meta, f"B_clause_index_oob({index}/{len(clauses)})"
        remaining = [c for i, c in enumerate(clauses) if i != index]
        if not remaining:
            return "", sel_meta, "B_would_delete_everything"
        # Join remaining clauses; add period at end if not present
        cf_response = ", ".join(remaining)
        if cf_response and cf_response[-1] not in ".?!;…":
            cf_response += "."
    else:
        return "", sel_meta, f"B_unknown_delete_unit({delete_unit})"

    # Final cleanup
    cf_response = re.sub(r"\s+", " ", cf_response).strip()
    return cf_response, sel_meta, ""


def build_cf_D_by_replace(
    response: str,
    selector: dict,
) -> Tuple[str, Dict[str, Any], str]:
    """Build counterfactual for bucket D by deterministic replace.

    Args:
        response: original response text
        selector: parsed selector JSON with target_text, replacement, type

    Returns:
        (cf_response, selector_meta, fail_reason)
        cf_response="" and fail_reason set if build fails.
    """
    target_text = selector.get("target_text", "")
    replacement = selector.get("replacement", "")
    span_type = selector.get("type", "")
    sel_meta = {
        "target_text": target_text,
        "replacement": replacement,
        "type": span_type,
    }

    # Abort signal
    if not target_text:
        return "", sel_meta, "selector_empty_target"

    # target_text must appear verbatim in response
    if target_text not in response:
        return "", sel_meta, "D_target_not_in_response"

    # replacement must differ from target
    if target_text == replacement:
        return "", sel_meta, "D_replacement_same_as_target"

    if not replacement:
        return "", sel_meta, "D_empty_replacement"

    # Check replacement token length <= 5
    repl_toks = _tokenize_simple(replacement)
    if len(repl_toks) > 5:
        return "", sel_meta, f"D_replacement_too_long({len(repl_toks)})"

    # Apply replace exactly once
    cf_response = response.replace(target_text, replacement, 1)

    # Guard: replacement must not introduce sentence-splitting punctuation
    orig_sents = len(split_sentences_vn(response))
    cf_sents = len(split_sentences_vn(cf_response))
    if cf_sents != orig_sents:
        return "", sel_meta, f"D_sentence_change_from_replace({cf_sents - orig_sents})"

    return cf_response, sel_meta, ""


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> Optional[dict]:
    """Try to extract the first valid JSON object from LLM output."""
    # Try direct parse first
    raw = raw.strip()
    # Remove possible markdown fences
    if raw.startswith("```"):
        # Remove opening fence
        idx = raw.find("\n")
        if idx != -1:
            raw = raw[idx + 1:]
        # Remove closing fence
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
        raw = raw.strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: find first { … last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Edit stats
# ---------------------------------------------------------------------------

def compute_edit_stats(original: str, counterfactual: str) -> Dict[str, Any]:
    """Compute token edit distance and sentence count change."""
    orig_toks = _tokenize_simple(original)
    cf_toks = _tokenize_simple(counterfactual)
    token_edits = _levenshtein_tokens(orig_toks, cf_toks)
    orig_sents = len(_sentence_split(original))
    cf_sents = len(_sentence_split(counterfactual))
    sentence_change = cf_sents - orig_sents
    deleted_tokens = len(orig_toks) - len(cf_toks)
    return {
        "token_edits": token_edits,
        "sentence_change": sentence_change,
        "orig_len_tokens": len(orig_toks),
        "cf_len_tokens": len(cf_toks),
        "orig_sentences": orig_sents,
        "cf_sentences": cf_sents,
        "deleted_tokens": deleted_tokens,
    }


# ---------------------------------------------------------------------------
# Deterministic safety computation
# ---------------------------------------------------------------------------

def compute_safety(
    context: str,
    response: str,
    cf_response: str,
    bucket: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute deterministic safety flags from (context, response, cf_response).

    No dependency on LLM self-report.  All flags are derived from
    token/string overlap heuristics so that validation is fully
    reproducible.

    Returns dict with:
        uses_only_context_facts   (bool)
        removed_unsupported       (bool)
        added_unsupported         (bool)
        introduced_contradiction  (bool)
    """

    # ---- normalise ----
    ctx_norm = _normalize_text(context)
    resp_norm = _normalize_text(response)
    cf_norm = _normalize_text(cf_response)

    ctx_tokens = set(_tokenize_simple(ctx_norm))
    resp_tokens = set(_tokenize_simple(resp_norm))
    cf_tokens_list = _tokenize_simple(cf_norm)
    cf_tokens = set(cf_tokens_list)
    resp_tokens_list = _tokenize_simple(resp_norm)

    # Novel tokens = in CF but not in response
    novel_tokens = cf_tokens - resp_tokens
    # Novel tokens not grounded in context
    novel_outside_context = novel_tokens - ctx_tokens

    # ==================== uses_only_context_facts ====================
    if bucket == "A":
        sent_ok = stats.get("sentence_change", 0) == 0
        novel_ok = len(novel_outside_context) <= NOVEL_OUTSIDE_CONTEXT_THRESHOLD
        orig_ov = _overlap_ratio(response, context)
        new_ov = _overlap_ratio(cf_response, context)
        overlap_ok = new_ov >= orig_ov - OVERLAP_DROP_THRESHOLD
        uses_only_context_facts = sent_ok and novel_ok and overlap_ok
    elif bucket == "B":
        # CF should be a subsequence → pure deletion, no new content
        # Use punctuation-stripped tokens for robust matching
        cf_stripped = _strip_punct_tokens(cf_tokens_list)
        resp_stripped = _strip_punct_tokens(resp_tokens_list)
        uses_only_context_facts = _is_token_subsequence(
            cf_stripped, resp_stripped
        )
    else:  # C, D purposely inject unsupported / contradiction
        uses_only_context_facts = False

    # ==================== removed_unsupported ====================
    if bucket == "B":
        cf_stripped = _strip_punct_tokens(cf_tokens_list)
        resp_stripped = _strip_punct_tokens(resp_tokens_list)
        removed_unsupported = _is_token_subsequence(
            cf_stripped, resp_stripped
        )
    else:
        removed_unsupported = False

    # ==================== added_unsupported ====================
    if bucket == "C":
        cf_sents = _sentence_split(cf_response)
        added = cf_sents[-1] if cf_sents else ""
        if added and len(_tokenize_simple(added)) >= MIN_ADDED_SENTENCE_TOKENS:
            in_context = _fuzzy_in(added, context)
            added_toks = set(_tokenize_simple(_normalize_text(added)))
            jaccard_val = _jaccard(added_toks, ctx_tokens)
            added_unsupported = (
                not in_context and jaccard_val < JACCARD_UNSUPPORTED_THRESHOLD
            )
        else:
            added_unsupported = False
    elif bucket in ("A", "B"):
        # Should NOT have added unsupported content
        added_unsupported = (
            len(novel_outside_context) > NOVEL_OUTSIDE_CONTEXT_THRESHOLD
        )
    else:
        added_unsupported = False

    # ==================== introduced_contradiction ====================
    if bucket == "D":
        # Use difflib on RAW (non-lowercased) tokens for entity/case detection
        orig_raw_toks = _tokenize_simple(response)
        cf_raw_toks = _tokenize_simple(cf_response)
        sm = difflib.SequenceMatcher(None, orig_raw_toks, cf_raw_toks)
        non_equal = [
            (t, i1, i2, j1, j2)
            for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t != "equal"
        ]
        if len(non_equal) == 1 and non_equal[0][0] == "replace":
            _, i1, i2, j1, j2 = non_equal[0]
            orig_span = orig_raw_toks[i1:i2]
            new_span = cf_raw_toks[j1:j2]
            orig_str = " ".join(orig_span)
            new_str = " ".join(new_span)

            has_digit = bool(re.search(r"\d", orig_str)) or bool(
                re.search(r"\d", new_str)
            )
            orig_has_neg = bool(
                VN_NEGATION_MARKERS & {t.lower() for t in orig_span}
            )
            new_has_neg = bool(
                VN_NEGATION_MARKERS & {t.lower() for t in new_span}
            )
            negation_flip = orig_has_neg != new_has_neg  # XOR

            entity_change = any(
                t[0].isupper() for t in orig_span if t
            ) or any(t[0].isupper() for t in new_span if t)

            introduced_contradiction = (
                has_digit or negation_flip or entity_change
            )
        else:
            introduced_contradiction = False
    else:
        introduced_contradiction = False

    return {
        "uses_only_context_facts": uses_only_context_facts,
        "removed_unsupported": removed_unsupported,
        "added_unsupported": added_unsupported,
        "introduced_contradiction": introduced_contradiction,
    }


# ---------------------------------------------------------------------------
# Validation (deterministic — no LLM safety_check dependency)
# ---------------------------------------------------------------------------

def validate_common(
    parsed: dict,
    response_orig: str,
    max_token_edits: int = MAX_TOKEN_EDITS,
    max_sentence_change: int = MAX_SENTENCE_CHANGE,
    bucket: str = "",
) -> Tuple[bool, str, Dict[str, Any]]:
    """Common validation: JSON structure + minimality + no-change check.

    Bucket B skips token_edits check (uses deleted_tokens in validate_bucket_B).
    """
    cf_resp = parsed.get("counterfactual_response", "")
    if not cf_resp:
        return False, "empty_counterfactual_response", {}

    # No-change check: normalized texts must differ
    if _normalize_for_compare(cf_resp) == _normalize_for_compare(response_orig):
        return False, "no_change", {}

    stats = compute_edit_stats(response_orig, cf_resp)

    # Minimality gate: sentence_change check applies to all buckets
    if abs(stats["sentence_change"]) > max_sentence_change:
        return False, f"minimality_fail_sent(sent={stats['sentence_change']})", stats
    # Token edits check: skip for bucket B (uses deleted_tokens metric instead)
    if bucket != "B" and stats["token_edits"] > max_token_edits:
        return False, f"minimality_fail_tok(tok={stats['token_edits']})", stats

    return True, "ok", stats


def validate_bucket_A(
    parsed: dict,
    response_orig: str,
    context: str,
    stats: dict,
    safety: dict,
) -> Tuple[bool, str]:
    """Bucket A (intrinsic->no): repair contradiction, use only context facts."""
    if stats.get("sentence_change", 0) != 0:
        return False, f"bucket_A_sentence_change({stats.get('sentence_change', 0)})"
    if not safety["uses_only_context_facts"]:
        return False, "bucket_A_uses_only_context_facts_false"
    if safety["added_unsupported"]:
        return False, "bucket_A_added_unsupported"
    return True, "ok"


def validate_bucket_B(
    parsed: dict,
    response_orig: str,
    stats: dict,
    safety: dict,
) -> Tuple[bool, str]:
    """Bucket B (extrinsic->no): deletion-only with deleted_tokens metric."""
    # Deletion-only: CF must be a subsequence of original
    if not safety["removed_unsupported"]:
        return False, "bucket_B_not_deletion"
    if safety["added_unsupported"]:
        return False, "bucket_B_added_unsupported"

    # Deleted tokens metric (replaces token_edits for B)
    deleted_tokens = stats.get("deleted_tokens",
        stats.get("orig_len_tokens", 0) - stats.get("cf_len_tokens", 0))
    if deleted_tokens < MIN_DELETED_TOKENS_B:
        return False, f"bucket_B_too_few_deleted({deleted_tokens})"
    if deleted_tokens > MAX_DELETED_TOKENS_B:
        return False, f"bucket_B_too_many_deleted({deleted_tokens})"

    # Sentence constraint
    orig_sents = stats.get("orig_sentences", 1)
    if orig_sents >= 2:
        if stats.get("sentence_change", 0) != -1:
            return False, f"bucket_B_sentence_change_not_minus1({stats.get('sentence_change', 0)})"
    else:
        # Single-sentence: clause deletion, sentence_change must be 0
        if stats.get("sentence_change", 0) != 0:
            return False, f"bucket_B_single_sent_change({stats.get('sentence_change', 0)})"
        if deleted_tokens < MIN_DELETED_TOKENS_B:
            return False, "bucket_B_single_sent_not_shorter"
    return True, "ok"


def validate_bucket_C(
    parsed: dict,
    response_orig: str,
    context: str,
    stats: dict,
    safety: dict,
) -> Tuple[bool, str]:
    """Bucket C (no->extrinsic): append exactly +1 sentence at end."""
    if stats.get("sentence_change", 0) != 1:
        return False, f"bucket_C_sentence_change_not_1({stats.get('sentence_change', 0)})"

    cf_resp = parsed["counterfactual_response"]

    # Enforce append-only: normalized CF must start with normalized original
    orig_norm = _normalize_for_compare(response_orig).rstrip(".?!;…")
    cf_norm_cmp = _normalize_for_compare(cf_resp)
    if not cf_norm_cmp.startswith(orig_norm):
        return False, "bucket_C_not_append_only"

    if not safety["added_unsupported"]:
        return False, "bucket_C_added_unsupported_false"
    if safety["introduced_contradiction"]:
        return False, "bucket_C_introduced_contradiction"

    # Validate appended sentence
    cf_sents = _sentence_split(cf_resp)
    added = cf_sents[-1] if cf_sents else ""
    if not added:
        return False, "bucket_C_no_added_sentence"
    added_toks = _tokenize_simple(added)
    if len(added_toks) < MIN_ADDED_SENTENCE_TOKENS:
        return False, f"bucket_C_added_sentence_too_short({len(added_toks)})"
    # Added sentence must not appear in context
    if _fuzzy_in(added, context):
        return False, "bucket_C_added_sentence_in_context"

    return True, "ok"


def validate_bucket_D(
    parsed: dict,
    response_orig: str,
    stats: dict,
    safety: dict,
) -> Tuple[bool, str]:
    """Bucket D (no->intrinsic): exactly one replace span, no sentence change."""
    # Sentence count unchanged
    if stats.get("sentence_change", 0) != 0:
        return False, f"bucket_D_sentence_change({stats.get('sentence_change', 0)})"
    # Token edits within strict bound
    if stats.get("token_edits", 999) > MAX_TOKEN_EDITS_D:
        return False, f"bucket_D_too_many_edits({stats.get('token_edits', 0)})"
    # Deterministic contradiction required
    if not safety["introduced_contradiction"]:
        return False, "bucket_D_no_contradiction"

    # Enforce exactly one local replace span via difflib
    cf_resp = parsed["counterfactual_response"]
    orig_toks = _tokenize_simple(response_orig)
    cf_toks = _tokenize_simple(cf_resp)
    sm = difflib.SequenceMatcher(None, orig_toks, cf_toks)
    non_equal_ops = [
        (tag, i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    ]
    if len(non_equal_ops) != 1:
        return False, f"bucket_D_multi_edit({len(non_equal_ops)}_ops)"
    if non_equal_ops[0][0] != "replace":
        return False, f"bucket_D_op_not_replace({non_equal_ops[0][0]})"

    return True, "ok"


def full_validate(
    parsed: dict,
    bucket: str,
    response_orig: str,
    context: str,
    max_token_edits: int = MAX_TOKEN_EDITS,
    max_sentence_change: int = MAX_SENTENCE_CHANGE,
) -> Tuple[bool, str, Dict[str, Any], Dict[str, Any]]:
    """Run common + bucket-specific validation with deterministic safety.

    Returns (ok, reason, stats, safety_dict).
    """
    # Use per-bucket max_token_edits if available
    bucket_max = BUCKET_MAX_TOKEN_EDITS.get(bucket, max_token_edits)

    ok, reason, stats = validate_common(
        parsed, response_orig, bucket_max, max_sentence_change, bucket=bucket
    )
    if not ok:
        return False, reason, stats, {}

    cf_resp = parsed.get("counterfactual_response", "")
    safety = compute_safety(context, response_orig, cf_resp, bucket, stats)

    if bucket == "A":
        ok2, reason2 = validate_bucket_A(
            parsed, response_orig, context, stats, safety
        )
    elif bucket == "B":
        ok2, reason2 = validate_bucket_B(parsed, response_orig, stats, safety)
    elif bucket == "C":
        ok2, reason2 = validate_bucket_C(
            parsed, response_orig, context, stats, safety
        )
    elif bucket == "D":
        ok2, reason2 = validate_bucket_D(parsed, response_orig, stats, safety)
    else:
        ok2, reason2 = False, f"unknown_bucket_{bucket}"

    if not ok2:
        return False, reason2, stats, safety

    return True, "ok", stats, safety


# ---------------------------------------------------------------------------
# HuggingFace generator
# ---------------------------------------------------------------------------

def load_generator(model_name: str, device: str = "auto", load_in_4bit: bool = False):
    """Load a HuggingFace causal LM + tokenizer for generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def make_messages(bucket: str, context: str, prompt: str, response: str, retry: bool = False) -> List[dict]:
    """Build chat messages for the generator.

    Bucket B/D use SYSTEM_PROMPT_SELECTOR (selector mode).
    Bucket A/C use SYSTEM_PROMPT (direct edit mode).
    """
    user_content = BUCKET_USER_TEMPLATES[bucket].format(
        context=context, prompt=prompt, response=response
    )
    if retry:
        if bucket == "B":
            user_content += RETRY_SUFFIX_B
        elif bucket == "D":
            user_content += RETRY_SUFFIX_D
        else:
            user_content += RETRY_SUFFIX
    sys_prompt = SYSTEM_PROMPT_SELECTOR if bucket in ("B", "D") else SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    bucket: str,
    retry: bool = False,
    seed: int = 42,
    max_new_tokens: int = 384,
) -> List[str]:
    """Generate counterfactuals one-by-one using HuggingFace generate().

    Processes items sequentially to stay safe on single-GPU setups.
    """
    torch.manual_seed(seed)
    temp = BUCKET_TEMP[bucket]

    results: List[str] = []
    for item in items:
        messages = make_messages(
            bucket, item["context"], item["prompt"], item["response"], retry=retry
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "top_p": 0.9,
            "do_sample": temp > 0,
        }
        if temp > 0:
            gen_kwargs["temperature"] = temp

        output_ids = model.generate(**enc, **gen_kwargs)
        gen_ids = output_ids[0, input_len:]
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(text_out)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def assign_bucket(label: str) -> List[str]:
    """Return bucket(s) applicable to this ground-truth label.
    Each sample participates in at most one "source" bucket.
    intrinsic -> bucket A
    extrinsic -> bucket B
    no        -> bucket C, D (two counterfactuals)
    """
    if label == "intrinsic":
        return ["A"]
    elif label == "extrinsic":
        return ["B"]
    elif label == "no":
        return ["C", "D"]
    return []


def load_progress(out_path: str) -> set:
    """Load already-processed (id, bucket) pairs from output JSONL."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add((obj["id"], obj["bucket"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ===========================================================================
# CONFIG — edit these variables directly (notebook-friendly, no argparse)
# ===========================================================================
TEST_PATH        = "./data/vihallu_test.csv"          # path to test CSV
OUT_DIR          = "./exp_out"                        # output directory
GENERATOR_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"     # HuggingFace model ID
BATCH_SIZE       = 16                                 # items per write-flush
SEED             = 42
LOAD_IN_4BIT     = True                               # 4-bit quantization
DEVICE           = "auto"                             # device map
CFG_MAX_TOKEN_EDITS   = MAX_TOKEN_EDITS               # general fallback
CFG_MAX_SENTENCE_CHG  = MAX_SENTENCE_CHANGE           # general fallback
NO_CLEAN         = False                              # skip text cleaning
BUCKETS_TO_RUN   = ["A", "B", "C", "D"]              # which buckets to run, e.g. ["A", "B"] or ["C", "D"]
RESUME_FROM      = None                               # path to JSONL to read progress from (None = same as output)
OUT_FILE         = "counterfactual_pairs.jsonl"        # output filename inside OUT_DIR

# ===========================================================================
# Pipeline execution
# ===========================================================================

# ---- resolve paths ----
test_path = Path(TEST_PATH)
out_dir   = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
out_path  = out_dir / OUT_FILE

# resume_path: file to read for progress check
# If RESUME_FROM is set, read progress from that file and write fresh to out_path.
# If RESUME_FROM is None, read and append to the same out_path (original behavior).
resume_path = Path(RESUME_FROM) if RESUME_FROM else out_path

print(f"[build_cf] Test path   : {test_path}")
print(f"[build_cf] Output file : {out_path}")
print(f"[build_cf] Resume from : {resume_path}")
print(f"[build_cf] Generator   : {GENERATOR_MODEL}")
print(f"[build_cf] Buckets     : {BUCKETS_TO_RUN}")

# ---- load test data ----
df = pd.read_csv(test_path)
print(f"[build_cf] Loaded {len(df)} test samples.")
print(f"[build_cf] Label distribution:\n{df['label'].value_counts().to_dict()}")

# Lightweight inline cleaner (NFC + collapse whitespace)
def _maybe_clean(text):
    if NO_CLEAN:
        return str(text)
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---- build work items ----
work_items: List[Dict[str, Any]] = []
for _, row in df.iterrows():
    label = row["label"]
    buckets = [b for b in assign_bucket(label) if b in BUCKETS_TO_RUN]
    for b in buckets:
        work_items.append({
            "id": row["id"],
            "context": _maybe_clean(row["context"]),
            "prompt": _maybe_clean(row["prompt"]),
            "response": _maybe_clean(row["response"]),
            "label": label,
            "bucket": b,
            "expected_label": BUCKET_MAP[b]["tgt"],
        })

print(f"[build_cf] Total work items: {len(work_items)}")
bucket_counts = {}
for wi in work_items:
    bucket_counts[wi["bucket"]] = bucket_counts.get(wi["bucket"], 0) + 1
print(f"[build_cf] Per bucket: {bucket_counts}")

# ---- check progress ----
done = load_progress(str(resume_path))
remaining = [wi for wi in work_items if (wi["id"], wi["bucket"]) not in done]
print(f"[build_cf] Already done: {len(done)}, remaining: {len(remaining)}")

# If RESUME_FROM is set and differs from out_path, copy existing valid records
# from resume_path to out_path so the new file is self-contained.
if RESUME_FROM and str(Path(RESUME_FROM).resolve()) != str(out_path.resolve()):
    copied = 0
    if resume_path.exists() and not out_path.exists():
        with open(resume_path, "r", encoding="utf-8") as rf, \
             open(out_path, "w", encoding="utf-8") as wf:
            for line in rf:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                    # Only copy records matching BUCKETS_TO_RUN
                    if obj.get("bucket", "") in BUCKETS_TO_RUN:
                        wf.write(line_s + "\n")
                        copied += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[build_cf] Copied {copied} existing records from resume file to {out_path}")
    elif not resume_path.exists():
        print(f"[build_cf] Resume file not found: {resume_path}, starting fresh.")
    else:
        print(f"[build_cf] Output file already exists, appending to {out_path}.")
        # Re-check done set from out_path to avoid duplicates
        done = done | load_progress(str(out_path))
        remaining = [wi for wi in work_items if (wi["id"], wi["bucket"]) not in done]
        print(f"[build_cf] After merging: already done: {len(done)}, remaining: {len(remaining)}")

if not remaining:
    print("[build_cf] Nothing to do. All items already generated.")
else:
    # ---- init generator model ----
    print(f"[build_cf] Loading generator model ({GENERATOR_MODEL})...")
    gen_model, gen_tokenizer = load_generator(
        GENERATOR_MODEL, device=DEVICE, load_in_4bit=LOAD_IN_4BIT
    )
    print("[build_cf] Generator model ready.")

    # ---- process per bucket ----
    for bucket_key in BUCKETS_TO_RUN:
        bucket_items = [wi for wi in remaining if wi["bucket"] == bucket_key]
        if not bucket_items:
            continue
        is_selector_mode = bucket_key in ("B", "D")
        mode_label = "[SELECTOR]" if is_selector_mode else "[DIRECT]"
        print(f"\n[build_cf] === Bucket {bucket_key} ({BUCKET_MAP[bucket_key]['src']}->{BUCKET_MAP[bucket_key]['tgt']}): {len(bucket_items)} items {mode_label} ===")

        # Process in batches
        for batch_start in range(0, len(bucket_items), BATCH_SIZE):
            batch = bucket_items[batch_start : batch_start + BATCH_SIZE]
            batch_end = min(batch_start + len(batch), len(bucket_items))
            print(f"  Batch {batch_start // BATCH_SIZE + 1}: items {batch_start+1}-{batch_end}/{len(bucket_items)}")

            # Generate (shorter output for selector mode)
            max_tokens = 128 if is_selector_mode else 384
            raw_outputs = generate_batch(
                gen_model, gen_tokenizer, batch, bucket_key,
                retry=False, seed=SEED, max_new_tokens=max_tokens,
            )

            # Validate + retry if needed
            retry_indices = []
            results = [None] * len(batch)

            for i, (item, raw) in enumerate(zip(batch, raw_outputs)):
                parsed = _extract_json(raw)
                if parsed is None:
                    retry_indices.append(i)
                    continue

                if is_selector_mode:
                    # Deterministic build from selector JSON
                    if bucket_key == "B":
                        cf_resp, sel_meta, build_reason = build_cf_B_by_deletion(
                            item["response"], parsed
                        )
                    else:  # D
                        cf_resp, sel_meta, build_reason = build_cf_D_by_replace(
                            item["response"], parsed
                        )
                    if not cf_resp:
                        results[i] = {
                            "parsed": {"counterfactual_response": ""},
                            "valid_cf": False,
                            "reason": build_reason or "deterministic_build_fail",
                            "stats": {},
                            "safety": {},
                            "retries": 0,
                            "raw": raw,
                            "selector_meta": sel_meta,
                        }
                        retry_indices.append(i)
                        continue
                    parsed_cf = {"counterfactual_response": cf_resp}
                else:
                    parsed_cf = parsed
                    sel_meta = {}

                ok, reason, stats, safety = full_validate(
                    parsed_cf, bucket_key, item["response"], item["context"],
                    CFG_MAX_TOKEN_EDITS, CFG_MAX_SENTENCE_CHG,
                )
                results[i] = {
                    "parsed": parsed_cf,
                    "valid_cf": ok,
                    "reason": reason,
                    "stats": stats,
                    "safety": safety,
                    "retries": 0,
                    "raw": raw,
                    "selector_meta": sel_meta,
                }
                if not ok:
                    retry_indices.append(i)

            # Retry pass
            if retry_indices:
                retry_batch = [batch[i] for i in retry_indices]
                retry_outputs = generate_batch(
                    gen_model, gen_tokenizer, retry_batch, bucket_key,
                    retry=True, seed=SEED + 1, max_new_tokens=max_tokens,
                )
                for ri, raw2 in zip(retry_indices, retry_outputs):
                    item = batch[ri]
                    parsed2 = _extract_json(raw2)
                    if parsed2 is None:
                        if results[ri] is None:
                            results[ri] = {
                                "parsed": None,
                                "valid_cf": False,
                                "reason": "json_parse_fail_after_retry",
                                "stats": {},
                                "safety": {},
                                "retries": 1,
                                "raw": raw2,
                                "selector_meta": {},
                            }
                        else:
                            results[ri]["retries"] = 1
                        continue

                    if is_selector_mode:
                        if bucket_key == "B":
                            cf_resp2, sel_meta2, build_reason2 = build_cf_B_by_deletion(
                                item["response"], parsed2
                            )
                        else:
                            cf_resp2, sel_meta2, build_reason2 = build_cf_D_by_replace(
                                item["response"], parsed2
                            )
                        if not cf_resp2:
                            if results[ri] is None or not results[ri]["valid_cf"]:
                                results[ri] = {
                                    "parsed": {"counterfactual_response": ""},
                                    "valid_cf": False,
                                    "reason": build_reason2 or "deterministic_build_fail_retry",
                                    "stats": {},
                                    "safety": {},
                                    "retries": 1,
                                    "raw": raw2,
                                    "selector_meta": sel_meta2,
                                }
                            continue
                        parsed_cf2 = {"counterfactual_response": cf_resp2}
                    else:
                        parsed_cf2 = parsed2
                        sel_meta2 = {}

                    ok2, reason2, stats2, safety2 = full_validate(
                        parsed_cf2, bucket_key, item["response"], item["context"],
                        CFG_MAX_TOKEN_EDITS, CFG_MAX_SENTENCE_CHG,
                    )
                    # Use retry result if better
                    if ok2 or results[ri] is None or not results[ri]["valid_cf"]:
                        results[ri] = {
                            "parsed": parsed_cf2,
                            "valid_cf": ok2,
                            "reason": reason2,
                            "stats": stats2,
                            "safety": safety2,
                            "retries": 1,
                            "raw": raw2,
                            "selector_meta": sel_meta2,
                        }

            # Write results
            with open(out_path, "a", encoding="utf-8") as fout:
                for item, res in zip(batch, results):
                    if res is None:
                        res = {
                            "parsed": None,
                            "valid_cf": False,
                            "reason": "generation_failed",
                            "stats": {},
                            "safety": {},
                            "retries": 0,
                            "raw": "",
                            "selector_meta": {},
                        }
                    record = {
                        "id": item["id"],
                        "context": item["context"],
                        "prompt": item["prompt"],
                        "response": item["response"],
                        "y_true": item["label"],
                        "counterfactual_response": (
                            res["parsed"].get("counterfactual_response", "")
                            if res["parsed"]
                            else ""
                        ),
                        "y_cf_expected": item["expected_label"],
                        "edit_stats": res["stats"],
                        "bucket": item["bucket"],
                        "valid_cf": res["valid_cf"],
                        "validation_reason": res["reason"],
                        "edit_summary": (
                            res["parsed"].get("edit_summary", []) if res["parsed"] else []
                        ),
                        "rationale_short": (
                            res["parsed"].get("rationale_short", "") if res["parsed"] else ""
                        ),
                        "safety_check": res.get("safety", {}),
                        "selector_meta": res.get("selector_meta", {}),
                        "gen_meta": {
                            "generator_model": GENERATOR_MODEL,
                            "temperature": BUCKET_TEMP[item["bucket"]],
                            "seed": SEED,
                            "retries": res["retries"],
                        },
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            valid_in_batch = sum(1 for r in results if r and r["valid_cf"])
            print(f"    -> valid: {valid_in_batch}/{len(batch)}")

# ---- summary & validation report ----
total_valid = 0
total = 0
reason_counter: Dict[str, Counter] = {b: Counter() for b in ["A", "B", "C", "D"]}
bucket_total: Dict[str, int] = {}
bucket_valid: Dict[str, int] = {}
bucket_B_deleted_tokens: List[int] = []  # for avg_deleted_tokens_B

# Report from the output file (contains all records for this run)
if os.path.exists(out_path):
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            bk = obj.get("bucket", "?")
            bucket_total[bk] = bucket_total.get(bk, 0) + 1
            if obj.get("valid_cf", False):
                total_valid += 1
                bucket_valid[bk] = bucket_valid.get(bk, 0) + 1
                # Collect deleted_tokens for valid B
                if bk == "B":
                    es = obj.get("edit_stats", {})
                    dt = es.get("deleted_tokens",
                        es.get("orig_len_tokens", 0) - es.get("cf_len_tokens", 0))
                    bucket_B_deleted_tokens.append(dt)
            else:
                reason_counter.setdefault(bk, Counter())[
                    obj.get("validation_reason", "unknown")
                ] += 1

print(f"\n[build_cf] DONE. Total pairs: {total}, valid: {total_valid}, coverage: {total_valid/max(total,1)*100:.1f}%")
print("[build_cf] Bucket-wise coverage:")
for bk in ["A", "B", "C", "D"]:
    bt = bucket_total.get(bk, 0)
    bv = bucket_valid.get(bk, 0)
    pct = bv / max(bt, 1) * 100
    print(f"  Bucket {bk} ({BUCKET_MAP[bk]['src']}->{BUCKET_MAP[bk]['tgt']}): {bv}/{bt} ({pct:.1f}%)")
print(f"[build_cf] Output: {out_path}")

# ---- write validation_report.json ----
report = {
    "total": total,
    "total_valid": total_valid,
    "coverage_pct": round(total_valid / max(total, 1) * 100, 2),
    "bucket_coverage": {},
    "invalid_reason_histogram": {},
}
for bk in ["A", "B", "C", "D"]:
    bt = bucket_total.get(bk, 0)
    bv = bucket_valid.get(bk, 0)
    report["bucket_coverage"][bk] = {
        "total": bt,
        "valid": bv,
        "coverage_pct": round(bv / max(bt, 1) * 100, 2),
    }
    report["invalid_reason_histogram"][bk] = dict(
        reason_counter.get(bk, Counter()).most_common()
    )

report_path = out_dir / "validation_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"[build_cf] Validation report: {report_path}")
print("Deterministic safety validation — paper-grade, reproducible.")
