"""Lines above a table (user, 2026-11-18: "the table import should also allow to skip
lines, that is something I so often had to do"): an instrument's header, a title, notes
are found and skipped; skip_rows says how many by hand."""
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from staplot import api  # noqa: E402


@pytest.fixture
def plate(tmp_path):
    path = tmp_path / "plate.csv"
    rng = np.random.default_rng(0)
    rows = "\n".join(",".join(f"{v:.2f}" for v in rng.normal(10, 1, 3)) for _ in range(8))
    path.write_text("Plate reader export v2.1\nDate: 2026-11-18, operator: XX\n\n# wells A1-H12\n"
                    "WT,mutA,mutB\n" + rows + "\n", encoding="utf-8")
    return str(path)


def test_an_instrument_header_is_skipped(plate):
    df = api.read_table(plate)
    assert list(df.columns) == ["WT", "mutA", "mutB"] and len(df) == 8
    assert df.attrs["skipped_lines"][0] == "Plate reader export v2.1"
    assert "3 line(s) above the table were skipped" in api.csv_cleaning_report(plate)["summary"]
    assert "skiprows=4" in api.plot_code(plate, "box")
    assert api.preamble_note(plate).startswith("the first 3 line(s)")


def test_skip_rows_by_hand(plate, tmp_path):
    assert list(api.read_table(plate, skip_rows=4).columns) == ["WT", "mutA", "mutB"]
    semi = tmp_path / "semi.txt"
    semi.write_text("title line\n\na;b\n1,5;2,5\n3,5;4,5\n", encoding="utf-8")
    df = api.read_table(str(semi), skip_rows=2)
    assert list(df.columns) == ["a", "b"] and df["a"].tolist() == [1.5, 3.5]
    t = api.table_from_csv(plate, skip_rows=4)
    assert [c.name for c in t.columns] == ["WT", "mutA", "mutB"]


def test_a_title_above_an_excel_table(tmp_path):
    pytest.importorskip("openpyxl")
    path = str(tmp_path / "titled.xlsx")
    rng = np.random.default_rng(1)
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([["Experiment 12"], ["operator: XX"]]).to_excel(w, header=False, index=False, sheet_name="s")
        pd.DataFrame({"WT": rng.normal(1, 1, 6), "KO": rng.normal(2, 1, 6)}).to_excel(
            w, startrow=3, index=False, sheet_name="s")
    df = api.read_table(path)
    assert list(df.columns) == ["WT", "KO"] and len(df) == 6
    assert df.attrs["skipped_lines"] == ["Experiment 12", "operator: XX"]
    code = api.plot_code(path, "box")
    exec(compile(code.replace('fig.savefig("plot.pdf")', ""), "c", "exec"), {})


def test_plain_tables_are_read_as_before(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("group,value\nA,1\nA,2\nB,3\nB,4\n", encoding="utf-8")
    df = api.read_table(str(path))
    assert list(df.columns) == ["group", "value"] and len(df) == 4 and not df.attrs["skipped_lines"]
