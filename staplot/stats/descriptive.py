import numpy as np
from scipy import stats


def describe(data: np.ndarray) -> dict:
    clean = data[~np.isnan(data)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan,
                "ci95": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan,
                "min": np.nan, "max": np.nan}
    mean = float(np.mean(clean))
    sd = float(np.std(clean, ddof=1)) if n > 1 else np.nan
    sem = float(stats.sem(clean)) if n > 1 else np.nan
    ci95 = float(stats.t.ppf(0.975, df=n - 1) * sem) if n > 1 else np.nan
    q1, median, q3 = (float(x) for x in np.percentile(clean, [25, 50, 75]))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95": ci95,
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def format_p(p: float) -> str:
    if np.isnan(p):
        return "p = n/a"
    if p < 0.0001:
        return "p < 0.0001"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def significance_stars(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
