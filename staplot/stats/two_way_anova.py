import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ..models.project import GroupedDataTable, AnalysisResult
from .descriptive import format_p, significance_stars


def run_two_way_anova(table: GroupedDataTable) -> AnalysisResult:
    df = table.to_long_dataframe()
    if df.empty:
        raise ValueError("No numeric data in table.")
    if df["factor_a"].nunique() < 2:
        raise ValueError("Need at least 2 levels for Factor A (columns).")

    model = smf.ols(
        "value ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)", data=df
    ).fit()
    anova_tbl = anova_lm(model, typ=2)

    fa_label = "Factor A (columns)"
    fb_label = "Factor B (rows)"

    def _fmt_row(label: str, row) -> str:
        f = row.get("F", np.nan)
        p = row["PR(>F)"]
        df_res = row.get("df", "")
        return (
            f"  {label:40s}  F = {f:8.3f}  {format_p(p):15s}  {significance_stars(p):4s}"
        )

    anova_lines = []
    row_labels = {
        "C(factor_a)": fa_label,
        "C(factor_b)": fb_label,
        "C(factor_a):C(factor_b)": "Interaction (A × B)",
    }
    for key, lbl in row_labels.items():
        if key in anova_tbl.index:
            anova_lines.append(_fmt_row(lbl, anova_tbl.loc[key]))

    # Tukey post-hoc for Factor A
    tukey_lines: list[str] = []
    if df["factor_a"].nunique() > 2:
        tukey = pairwise_tukeyhsd(df["value"], df["factor_a"])
        tukey_lines = ["", "Post-hoc Factor A (Tukey HSD):", str(tukey.summary())]

    # Cell means table
    means = df.groupby(["factor_a", "factor_b"])["value"].agg(["mean", "std", "count"])
    cell_header = f"\n{'':5s}" + "".join(f"{c:>12s}" for c in table.col_groups)
    cell_lines = [cell_header]
    for rl in table.row_labels:
        row_str = f"{rl:5s}"
        for cg in table.col_groups:
            try:
                m = means.loc[(cg, rl), "mean"]
                row_str += f"{m:>12.4f}"
            except KeyError:
                row_str += f"{'—':>12s}"
        cell_lines.append(row_str)

    lines = [
        "=== Two-Way ANOVA ===",
        f"Table: {table.name}",
        f"Factor A (columns): {', '.join(table.col_groups)}",
        f"Factor B (rows):    {', '.join(table.row_labels)}",
        "",
        "ANOVA Table (Type II SS):",
        *anova_lines,
        "",
        "Cell Means:",
        *cell_lines,
        *tukey_lines,
    ]
    return AnalysisResult(
        name=f"Two-Way ANOVA: {table.name}",
        method="two-way-anova",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={
            "fa_p": float(anova_tbl.loc["C(factor_a)", "PR(>F)"]) if "C(factor_a)" in anova_tbl.index else np.nan,
            "fb_p": float(anova_tbl.loc["C(factor_b)", "PR(>F)"]) if "C(factor_b)" in anova_tbl.index else np.nan,
            "interaction_p": float(anova_tbl.loc["C(factor_a):C(factor_b)", "PR(>F)"])
            if "C(factor_a):C(factor_b)" in anova_tbl.index
            else np.nan,
        },
    )
