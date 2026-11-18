# staplot

Statistical analysis and publication-quality plots for biologists: a free,
open-source alternative to GraphPad Prism, with a desktop app and a small Python API.

## Features

- **Data tables** Prism-style (one column per group) or long format; CSV and Excel
  import (semicolon / decimal-comma CSVs included).
- **Tests:** t-tests (Welch, paired), Mann-Whitney, Wilcoxon, one- and two-way ANOVA,
  repeated measures, Kruskal-Wallis, post-hoc tests (Tukey, Dunn, ...), chi-square,
  correlation, linear and nonlinear regression (curve fitting), normality tests,
  outliers, ROC, survival, power analysis.
- **Plots:** bar, box, violin (whole or half: `violin_side="left"`), split violin (`kind="split_violin"`: two conditions as the two halves, each pair tested), scatter, grouped bar, before/after, histogram, QQ,
  residuals, Bland-Altman, heatmap, volcano, ROC and survival curves, with
  significance brackets added automatically.
- A test wizard that picks the right test for your data and explains why.

## Install

```
pip install staplot            # add [excel] for .xlsx / .xls import
```

Python 3.10 or newer.

## Use

Desktop app:

```
staplot
```

From Python (no GUI needed, safe to run headless):

```python
from staplot import api

table = api.table_from_csv("counts.csv")       # one column per group
stats = api.auto_test(table)                   # picks the test, returns p-values
fig, record = api.plot(table, kind="box")      # matplotlib Figure with brackets
fig.savefig("counts.pdf")
```

Tables can be CSV / TSV / TXT (separator and decimal comma detected) or Excel
sheets: `api.table_from_csv("book.xlsx::counts")` reads the sheet *counts* (by name
or number from 1; the first sheet without `::`). `api.read_table(path)` gives the
DataFrame, `api.list_sheets(path)` the sheet names, and
`api.csv_cleaning_report(path)` what was left out (empty cells, NA, typos).

`auto_test` checks normality per group (Shapiro-Wilk) and chooses Welch's t-test or
Mann-Whitney for two groups, and one-way ANOVA + Tukey or Kruskal-Wallis + Dunn for
more.

staplot draws the plot panels of [PyFigures](https://github.com/baigouy/PyFigures).

## Tests

```
pip install pytest pytest-qt
python -m pytest tests
```

## License

BSD 2-Clause, see [LICENSE](LICENSE).
