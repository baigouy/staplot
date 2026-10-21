"""ROC curve computation and AUC (no sklearn dependency), and the log-rank test."""

import numpy as np
from scipy import stats

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p, significance_stars


def validate_binary(values, name: str = "label") -> np.ndarray:
    """Exact 0/1 values as ints; rejects fractions, other numbers and NaN
    BEFORE any cast (0.5 must not silently become 0)."""
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite 0/1.")
    bad = sorted(set(arr[(arr != 0) & (arr != 1)].tolist()))
    if bad:
        raise ValueError(f"{name} values must be exactly 0 or 1 (found {bad[:5]}).")
    return arr.astype(int)


def roc_curve(labels: np.ndarray, scores: np.ndarray):
    """(fpr, tpr, thresholds, auc) with ONE curve point per distinct score, so
    tied scores move the curve diagonally (half credit) whatever their order.
    thresholds[i] is the cut (score >= threshold → positive) of point i; the
    first point is (0, 0) at threshold +inf."""
    labels = validate_binary(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both classes (at least one 0 and one 1).")
    thr = np.unique(scores)[::-1]
    tp = np.array([np.sum((scores >= t) & (labels == 1)) for t in thr], dtype=float)
    fp = np.array([np.sum((scores >= t) & (labels == 0)) for t in thr], dtype=float)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    thresholds = np.concatenate([[np.inf], thr])
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapz(tpr, fpr))
    return fpr, tpr, thresholds, auc


def compute_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (fpr, tpr, auc).
    labels: 0/1 array   scores: continuous predictor (higher = more positive)
    """
    fpr, tpr, _, auc = roc_curve(labels, scores)
    return fpr, tpr, auc


def _delong_variance(labels: np.ndarray, scores: np.ndarray) -> float:
    """SE of the AUC (DeLong placement values, ties = half credit). NaN when a
    class has fewer than 2 observations (its variance is undefined)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return float("nan")
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    V10 = psi.mean(axis=1)
    V01 = psi.mean(axis=0)
    s10 = np.var(V10, ddof=1) / n1
    s01 = np.var(V01, ddof=1) / n2
    return float(np.sqrt(s10 + s01))


def run_roc(
    table: DataTable,
    label_col: str,
    score_col: str,
) -> AnalysisResult:
    # label and score of a subject come from the same row
    labels, scores = table.paired_arrays(label_col, score_col)
    labels = validate_binary(labels, f"Label column '{label_col}'")

    fpr, tpr, thresholds, auc = roc_curve(labels, scores)
    se = _delong_variance(labels, scores)
    if np.isfinite(se) and se > 0:
        z = (auc - 0.5) / se
        p = float(2 * stats.norm.sf(abs(z)))
        ci_lo = max(0.0, auc - 1.96 * se)
        ci_hi = min(1.0, auc + 1.96 * se)
    else:
        z = p = ci_lo = ci_hi = float("nan")

    # Optimal threshold (Youden index), read at the same curve point
    youden = tpr - fpr
    opt_idx = int(np.argmax(youden))
    opt_thresh = float(thresholds[opt_idx])

    n_pos, n_neg = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    lines = [
        "=== ROC Curve Analysis ===",
        f"Label column: {label_col}  (0=negative, 1=positive)",
        f"Score column: {score_col}",
        f"N = {len(labels)}  (positive={n_pos}, negative={n_neg})",
        "",
        f"AUC           = {auc:.4f}",
    ]
    if np.isfinite(se):
        lines += [
            f"95% CI        = [{ci_lo:.4f}, {ci_hi:.4f}]",
            f"SE            = {se:.4f}",
            f"z             = {z:.4f}",
            f"{format_p(p)}  {significance_stars(p)}",
        ]
    else:
        lines.append("SE / CI / p: not computed (a class has fewer than 2 observations)")
    lines += [
        "",
        f"Optimal threshold (Youden index) ≈ {opt_thresh:.4f}",
        "  (score ≥ threshold → positive; maximises sensitivity + specificity − 1)",
    ]
    return AnalysisResult(
        name=f"ROC: {score_col} predicting {label_col}",
        method="roc",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={
            "auc": auc, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi, "se": se,
            "label_col": label_col, "score_col": score_col,
            "threshold": opt_thresh,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": [None if not np.isfinite(t) else float(t) for t in thresholds],
        },
    )


def log_rank(t1, e1, t2, e2) -> tuple[float, float, dict]:
    """Two-group log-rank test (Mantel-Cox) with the hypergeometric risk-set
    variance V = Σ n1·n2·d·(n−d) / (n²·(n−1)). Returns (chi2, p, counts)."""
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    e1, e2 = validate_binary(e1, "event"), validate_binary(e2, "event")
    if not (np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))):
        raise ValueError("times must be finite.")
    event_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1 = E1 = V = 0.0
    for t in event_times:
        n1 = float(np.sum(t1 >= t))
        n2 = float(np.sum(t2 >= t))
        d1 = float(np.sum((t1 == t) & (e1 == 1)))
        d2 = float(np.sum((t2 == t) & (e2 == 1)))
        n, d = n1 + n2, d1 + d2
        O1 += d1
        E1 += n1 * d / n
        if n > 1:
            V += n1 * n2 * d * (n - d) / (n * n * (n - 1))
    if V > 0:
        chi2 = (O1 - E1) ** 2 / V
        p = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2, p = float("nan"), float("nan")
    return float(chi2), p, {"O1": O1, "E1": E1, "V": V,
                            "O2": float(np.sum(e2 == 1)),
                            "E2": float(np.sum(e1 == 1) + np.sum(e2 == 1)) - E1}


def run_log_rank(
    table: DataTable,
    time_col_a: str,
    event_col_a: str,
    time_col_b: str,
    event_col_b: str,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> AnalysisResult:
    """Log-rank test to compare two survival curves."""
    # a subject's time and event flag come from the same row
    t1, e1 = table.paired_arrays(time_col_a, event_col_a)
    t2, e2 = table.paired_arrays(time_col_b, event_col_b)
    chi2, p, cnt = log_rank(t1, e1, t2, e2)
    e1, e2 = validate_binary(e1, "event"), validate_binary(e2, "event")

    lines = [
        "=== Log-Rank Test ===",
        f"Group A: {label_a}  (n={len(t1)}, events={int(np.sum(e1 == 1))})",
        f"Group B: {label_b}  (n={len(t2)}, events={int(np.sum(e2 == 1))})",
        "",
    ]
    if np.isfinite(chi2):
        lines += [f"χ²(1) = {chi2:.4f}", f"{format_p(p)}  {significance_stars(p)}"]
    else:
        lines.append("χ² not defined (no variance: no informative event times).")
    return AnalysisResult(
        name=f"Log-rank: {label_a} vs {label_b}",
        method="log-rank",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"chi2": chi2, "p": p, **cnt},
    )
