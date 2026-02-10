# -*- coding: utf-8 -*-
"""Inspecte les colonnes des fichiers élections 2012 / 2022."""
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BRONZE = PROJECT / "bronze"

def main():
    out = []
    # 2012 présidentielle
    p = BRONZE / "presidentielleResultats2012.xls"
    if p.exists():
        df0 = pd.read_excel(p, header=0)
        out.append(("2012_pres_h0", list(df0.columns)[:35], df0.shape))
        # colonnes contenant Voix ou %
        voix_cols = [c for c in df0.columns if "Voix" in str(c) or (str(c).startswith("%") and "Exp" in str(c))]
        out.append(("2012_pres_voix_like", voix_cols[:20], None))
        df3 = pd.read_excel(p, header=3)
        out.append(("2012_pres_h3", list(df3.columns)[:18], df3.shape))
    # 2012 législatives
    p = BRONZE / "legislativesResultat2012.xls"
    if p.exists():
        df = pd.read_excel(p, header=0)
        out.append(("2012_legislative", list(df.columns)[:30], df.shape))
    # 2022 (subcom = commune)
    p = BRONZE / "resultats-par-niveau-subcom-t1-france-entiere2022.xlsx"
    if p.exists():
        xl = pd.ExcelFile(p)
        out.append(("2022_sheets", xl.sheet_names, None))
        df = pd.read_excel(p, sheet_name=0, nrows=5)
        out.append(("2022_subcom_cols", list(df.columns), df.shape))
    for name, cols_or_sheets, shape in out:
        print(name, shape)
        print(cols_or_sheets[:30] if isinstance(cols_or_sheets[0], str) else cols_or_sheets)

if __name__ == "__main__":
    main()
