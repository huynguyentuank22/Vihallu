# Metric Formulas — UIT-ViHallu

## 1. Standard Classification Metrics

### Accuracy

$$
\text{Accuracy} = \frac{\sum_{c} \text{TP}_c}{N}
$$

### Precision, Recall, F1 (per class $c$)

$$
\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}
\qquad
\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
$$

$$
\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

### Macro-F1 (3 classes: `no`, `intrinsic`, `extrinsic`)

$$
\text{F1}_{\text{macro}} = \frac{1}{|C|} \sum_{c \in C} \text{F1}_c
$$

### Weighted-F1

$$
\text{F1}_{\text{weighted}} = \frac{\sum_{c \in C} \text{support}_c \cdot \text{F1}_c}{\sum_{c \in C} \text{support}_c}
$$

---

## 2. Counterfactual Evaluation Metrics

Ký hiệu:
- $\mathcal{V}$ = tập counterfactual hợp lệ (`valid_cf = True`), $n = |\mathcal{V}|$
- $\hat{y}_i^{\text{orig}}$ = prediction trên sample gốc
- $\hat{y}_i^{\text{cf}}$ = prediction trên counterfactual
- $y_i^{\text{cf,exp}}$ = expected label sau khi edit (theo bucket)
- $p_i^{\text{orig}}[\cdot]$, $p_i^{\text{cf}}[\cdot]$ = xác suất softmax trên sample gốc / CF

### CFR — Counterfactual Flip Rate

> Tỉ lệ CF mà model dự đoán đúng label mong đợi sau khi edit.

$$
\text{CFR} = \frac{1}{n} \sum_{i \in \mathcal{V}} \mathbf{1}\!\left[\hat{y}_i^{\text{cf}} = y_i^{\text{cf,exp}}\right]
$$

### DCS — Decision Change Score

> Trong số các sample mà prediction **thực sự thay đổi**, tỉ lệ flip đúng hướng.

$$
\mathcal{F} = \left\{i \in \mathcal{V} \mid \hat{y}_i^{\text{cf}} \neq \hat{y}_i^{\text{orig}}\right\}
$$

$$
\text{DCS} = \frac{1}{|\mathcal{F}|} \sum_{i \in \mathcal{F}} \mathbf{1}\!\left[\hat{y}_i^{\text{cf}} = y_i^{\text{cf,exp}}\right]
$$

### ΔConf — Delta Confidence

> Mức tăng confidence về expected label sau edit, so với confidence về predicted label trước edit.

$$
\Delta\text{Conf}_i = p_i^{\text{cf}}\!\left[y_i^{\text{cf,exp}}\right] - p_i^{\text{orig}}\!\left[\hat{y}_i^{\text{orig}}\right]
$$

$$
\overline{\Delta\text{Conf}} = \frac{1}{n} \sum_{i \in \mathcal{V}} \Delta\text{Conf}_i
$$

Cũng báo cáo median: $\text{median}_{i \in \mathcal{V}}(\Delta\text{Conf}_i)$

### MESR@k — Minimal-Edit Success Rate

> CFR giới hạn trên tập CF có kích thước edit $\leq k$ token.

$$
\mathcal{V}_k = \left\{i \in \mathcal{V} \mid \text{edit\_size}_i \leq k\right\}
$$

$$
\text{MESR}@k = \frac{1}{|\mathcal{V}_k|} \sum_{i \in \mathcal{V}_k} \mathbf{1}\!\left[\hat{y}_i^{\text{cf}} = y_i^{\text{cf,exp}}\right]
$$

với $k \in \{1, 3, 5, 10\}$.

### Coverage Rate

$$
\text{Coverage} = \frac{n}{|\mathcal{V}_{\text{total}}|}
$$

---

## 3. Stability Metrics

Ký hiệu:
- $\mathcal{S}$ = tập samples được áp dụng non-semantic edit (swap câu / thêm dấu chấm)
- $r_i^{\text{ns}}$ = prediction trên non-semantic variant

### Flip Rate (Non-semantic)

$$
\text{flip\_rate} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \mathbf{1}\!\left[r_i^{\text{ns}} \neq r_i^{\text{orig}}\right]
$$

### Stability Score (SS)

$$
\text{SS} = 1 - \text{flip\_rate}
$$

### Sensitivity Gap (SG)

$$
\text{SG} = \text{CFR}_{\text{semantic}} - \text{flip\_rate}_{\text{nonsemantic}}
$$

SG cao → model phân biệt được semantic vs non-semantic changes.

---

## 4. Edit Statistics (dùng trong Validation & MESR)

### Token Edit Distance (Levenshtein trên token)

$$
\text{token\_edits} = \text{Levenshtein}\!\left(\text{tokens}(r),\, \text{tokens}(r^{\text{cf}})\right)
$$

### Sentence Change

$$
\text{sentence\_change} = \left|\text{sents}(r^{\text{cf}})\right| - \left|\text{sents}(r)\right|
$$

### Deleted Tokens

$$
\text{deleted\_tokens} = \left|\text{tokens}(r)\right| - \left|\text{tokens}(r^{\text{cf}})\right|
$$

### Edit Size (unified, dùng cho MESR)

$$
\text{edit\_size} = \begin{cases}
\text{token\_edits} & \text{nếu tồn tại} \\
\text{deleted\_tokens} & \text{fallback (Bucket B)} \\
+\infty & \text{nếu thiếu metadata}
\end{cases}
$$

---

## 5. Bucket Validation Thresholds

| Bucket | Chuyển đổi | Max `token_edits` | Constraint `sentence_change` |
|--------|------------|-------------------|------------------------------|
| A | intrinsic → no | ≤ 25 | 0 (giữ nguyên số câu) |
| B | extrinsic → no | — | −1 nếu response có ≥2 câu; 0 nếu chỉ có 1 câu (xóa mệnh đề) |
| B | (deleted\_tokens) | ∈ [3, 80] | — |
| C | no → extrinsic | ≤ 15 | +1 (thêm đúng 1 câu) |
| D | no → intrinsic | ≤ 10 | 0 (giữ nguyên số câu) |

---

## 6. No-Context Ablation

Tất cả metrics ở Mục 2 được tính lại với `context = ""`.

CFR drop per bucket đo lường mức độ phụ thuộc vào context:

$$
\text{CFR\_drop}^{(b)} = \text{CFR}^{(b)} - \text{CFR}_{\text{no-context}}^{(b)}, \quad b \in \{A, B, C, D\}
$$
