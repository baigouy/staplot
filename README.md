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

Plot editor: a table beside its plot; click any part of the plot (a group, the
points, the brackets, a title, an axis label, the tick labels, the legend, the plot
area) to change its colour, text, size or font; edit the table's cells and the plot
follows. Toolbar *Edit a plot* in the app, or:

```
python -m staplot.ui.plot_editor counts.csv box
```

From Python: `staplot.ui.plot_editor.PlotEditor(path, kind, kwargs)`; after OK,
`kwargs()` / `changed_kwargs()` are `api.plot`'s arguments and `text_parts()` the size
and font per part of the text. PyFigures opens it on a double-click on a data plot.
Icons come from `qtawesome` when it is installed (optional).

Beyond the basic kinds, `api.plot` takes `layers` (mean / median lines, mean ± error over
the groups), `y2` (a measure on a right-hand axis), `invert_y` / `invert_x`, `ybreak`
(a broken y axis), `y_step` / `y_format` / `log_base`, `error_side="up"`, and the kinds
`stacked` (`percent`), `dose_response` (4-parameter logistic fit, EC50), `histogram`,
`heatmap`. `api.table_from_csv(path, columns=[group, value], subject="animal")` makes the
animals the replicates (tests on their means, a SuperPlot, plus a mixed model).
`api.image_plot(path, pixel_size=0.2, unit="µm")` draws an image with its coordinate axes.

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
