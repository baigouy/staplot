"""Tables from CSV / TXT / Excel (user: "most users may want to pass an excel sheet",
"by xls I mean xls and xlsx")."""
import numpy as np
import pandas as pd
import pytest

from staplot import api


def test_a_typo_in_a_decimal_comma_column_keeps_the_group(tmp_path):
    """French Excel's ';' + '41,2': one 'abc' kept the whole column as text, and the
    group vanished from the plot."""
    p = tmp_path / "fr.txt"
    p.write_text("WT;mutB\n41,2;40,3\n40,1;abc\n42,5;38,3\n", encoding="utf-8")
    df = api.read_table(str(p))
    assert pd.to_numeric(df["mutB"], errors="coerce").notna().sum() == 2
    report = api.csv_cleaning_report(str(p))
    assert "mutB: n=2" in report["summary"] and "'abc'" in report["summary"]


def test_a_thousands_comma_is_not_read_as_a_decimal_in_a_dot_table(tmp_path):
    p = tmp_path / "en.csv"
    p.write_text('a,b\n1.5,"1,234"\n2.5,"2,345"\n', encoding="utf-8")
    df = api.read_table(str(p))
    assert df["a"].tolist() == [1.5, 2.5]
    assert pd.to_numeric(df["b"], errors="coerce").isna().all()       # not 1.234


@pytest.fixture
def book(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "m.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"WT": [1.0, 2.0, 3.0], "mut": [4.0, 5.0, 6.0]}).to_excel(
            w, sheet_name="counts", index=False, startrow=1, startcol=1)        # blank margin
        pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}).to_excel(w, sheet_name="tidy", index=False)
    from openpyxl import load_workbook
    wb = load_workbook(path)
    wb["counts"]["C4"].value = "5,5"                                         # a number typed as text
    wb.save(path)
    return str(path)


def test_an_excel_sheet_is_read_by_name_or_number_and_its_margin_dropped(book):
    assert api.list_sheets(book) == ["counts", "tidy"]
    first = api.read_table(book)
    assert list(first.columns) == ["WT", "mut"]
    assert first["WT"].tolist() == [1.0, 2.0, 3.0] and first["mut"].tolist() == [4.0, 5.5, 6.0]
    assert list(api.read_table(book + "::tidy").columns) == ["g", "v"]
    assert list(api.read_table(book + "::2").columns) == ["g", "v"]
    with pytest.raises(ValueError, match="its sheets are counts, tidy"):
        api.read_table(book + "::nope")


def test_statistics_run_on_an_excel_sheet(book):
    table = api.table_from_csv(book + "::counts")
    assert table is not None
    assert api.csv_cleaning_report(book + "::counts")["summary"].endswith("no value left out.")


def test_an_old_xls_needs_xlrd_and_says_so(tmp_path, monkeypatch):
    p = tmp_path / "old.xls"
    p.write_bytes(b"not really")
    import builtins
    real = builtins.__import__

    def no_xlrd(name, *a, **k):
        if name == "xlrd":
            raise ImportError("no xlrd")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_xlrd)
    with pytest.raises(ImportError, match="pip install xlrd"):
        api.read_table(str(p))
