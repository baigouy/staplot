"""Missing and non-numeric cells (2026-11-18): left out group by group, never imputed, and
reported (csv_cleaning_report) so the plot's reader, or a language model, can say what
was done. A long-table row without a group label is not a group named "nan"."""
from staplot import api


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_wide_table_leaves_out_empty_na_and_text_cells_and_says_so(tmp_path):
    path = _write(tmp_path, "wide.csv", "WT,mutA,mutB\n42,33,45\n40,,44\n45,NA,nan\n39,35,46\n"
                                        "41,n/a,43\n44,31,\n38,abc,47\n")
    groups = api.table_from_csv(path).group_arrays()
    assert {k: len(v) for k, v in groups.items()} == {"WT": 7, "mutA": 3, "mutB": 5}
    report = api.csv_cleaning_report(path)
    assert report["layout"] == "wide"
    assert report["groups"]["mutA"] == {"used": 3, "missing": 3, "non_numeric": 1, "non_numeric_examples": ["abc"]}
    assert report["groups"]["WT"]["missing"] == 0 and report["groups"]["mutB"]["missing"] == 2
    assert "'abc'" in report["summary"] and "never filled in" in report["summary"]


def test_long_table_row_without_group_is_left_out_not_a_nan_group(tmp_path):
    path = _write(tmp_path, "long.csv", "genotype,value\nWT,42\nWT,\nWT,40\nmutA,NA\nmutA,33\nmutA,35\n"
                                        "mutB,45\nmutB,nan\nmutB,44\n,41\n  ,39\n")
    groups = api.table_from_csv(path).group_arrays()
    assert set(groups) == {"WT", "mutA", "mutB"}
    report = api.csv_cleaning_report(path)
    assert report["layout"] == "long" and report["rows_without_group"] == 2
    assert all(g["used"] == 2 and g["missing"] == 1 for g in report["groups"].values())


def test_a_clean_table_reports_nothing_left_out(tmp_path):
    path = _write(tmp_path, "clean.csv", "A,B\n1,2\n2,3\n3,4\n")
    report = api.csv_cleaning_report(path)
    assert report["summary"].endswith("no value left out.")
    assert all(g["missing"] == 0 and g["non_numeric"] == 0 for g in report["groups"].values())
