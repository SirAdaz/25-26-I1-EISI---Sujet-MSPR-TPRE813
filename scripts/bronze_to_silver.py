# -*- coding: utf-8 -*-
"""
Pipeline Bronze → Silver : lecture des données brutes, nettoyage et filtrage
par année/élection. Sortie : fichiers Silver par thème (élections, géographie,
état civil) avec code commune standardisé (CODGEO).

Structure du script (pour s'y retrouver) :
  1. Utilitaires : code_geo, noms de colonnes, extraction voix
  2. Élections : 2017 (principale), 2012 pres/leg, 2022 leg
  3. Géographie : population + indicateurs par commune/année
  4. État civil : mariages, décès, naissances (département ou commune)
  5. Emploi / chômage : enquête emploi par année, chômage par département
  6. Autres : circo, population dep, Filosofi, diplômes, population communale
  7. Run : enchaînement des process_* et écriture Silver
"""
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# -----------------------------------------------------------------------------
# Constantes (noms de colonnes Excel élections)
# -----------------------------------------------------------------------------
_COL_CODE_DEP = "Code du département"
_COL_CODE_COM = "Code de la commune"
_COL_LIBELLE_COM = "Libellé de la commune"
_COL_LIBELLE_DEP = "Libellé du département"


# -----------------------------------------------------------------------------
# Utilitaires : répertoire Silver, code commune Insee, noms de colonnes
# -----------------------------------------------------------------------------

def ensure_silver_dir():
    config.SILVER_DIR.mkdir(parents=True, exist_ok=True)


def normalize_codgeo(code_dep, code_com) -> str:
    """Construit le code commune Insee (5 caractères) à partir du département et de la commune."""
    if pd.isna(code_dep) or pd.isna(code_com):
        return ""
    code_dep = str(code_dep).strip()
    try:
        code_com = str(int(float(code_com))).zfill(3)
    except (ValueError, TypeError):
        code_com = str(code_com).strip().zfill(3)
    if code_dep in ("2A", "2a"):
        return "2A" + code_com
    if code_dep in ("2B", "2b"):
        return "2B" + code_com
    try:
        return str(int(float(code_dep))).zfill(2) + code_com
    except (ValueError, TypeError):
        return (code_dep.zfill(2) if len(code_dep) <= 2 else code_dep) + code_com


def _normalize_col_names(columns):
    """Normalise les noms de colonnes (accents, apostrophes)."""
    return [
        str(c).replace("é", "e").replace("è", "e").replace("ô", "o").replace("'", "_").replace("â", "a")
        for c in columns
    ]


def _extract_voix_columns_wide(df, first_row, max_candidates=25):
    """Extrait les colonnes voix_<candidat> depuis un format large (Nom, Voix ou Nom.1, Voix.1...)."""
    out = {}
    for i in range(max_candidates):
        nom_col = "Nom" if i == 0 else f"Nom.{i}"
        voix_col = "Voix" if i == 0 else f"Voix.{i}"
        if voix_col not in df.columns:
            continue
        nom_val = first_row.get(nom_col, "")
        if pd.isna(nom_val) or not str(nom_val).strip():
            continue
        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", str(nom_val).strip())[:50]
        out[f"voix_{safe_name}"] = pd.to_numeric(df[voix_col], errors="coerce").astype("Int64")
    return out


# -----------------------------------------------------------------------------
# Élections : résultats par commune (format wide = une ligne par commune)
# -----------------------------------------------------------------------------

def process_elections() -> pd.DataFrame:
    """Lit le XLS des résultats élection 2017, nettoie et retourne un DataFrame Silver."""
    if not config.BRONZE_ELECTIONS_XLS.exists():
        print(f"Fichier introuvable: {config.BRONZE_ELECTIONS_XLS}")
        return pd.DataFrame()

    df = pd.read_excel(config.BRONZE_ELECTIONS_XLS, header=3)  # ligne 4 = en-têtes
    df = df.rename(columns={
        _COL_CODE_DEP: "code_dep",
        _COL_LIBELLE_DEP: "libelle_dep",
        _COL_CODE_COM: "code_com",
        _COL_LIBELLE_COM: "libelle_commune",
        "Inscrits": "inscrits",
        "Votants": "votants",
        "Exprimés": "exprimes",
    })
    df.columns = _normalize_col_names(df.columns)
    df["code_geo"] = df.apply(
        lambda r: normalize_codgeo(r.get("code_dep"), r.get("code_com")),
        axis=1
    )
    base_cols = ["code_geo", "code_dep", "libelle_dep", "libelle_commune", "inscrits", "votants", "exprimes"]
    available_base = [c for c in base_cols if c in df.columns]
    silver_elections = df[available_base].copy()
    for col_name, series in _extract_voix_columns_wide(df, df.iloc[0]).items():
        silver_elections[col_name] = series
    return silver_elections


def _read_elections_xls_2012(path) -> pd.DataFrame:
    """Lit un XLS résultats 2012 (header=0, même structure Nom/Voix.1 que 2017)."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_excel(path, header=0)
    df = df.rename(columns={
        _COL_CODE_DEP: "code_dep",
        _COL_LIBELLE_DEP: "libelle_dep",
        _COL_CODE_COM: "code_com",
        _COL_LIBELLE_COM: "libelle_commune",
        "Inscrits": "inscrits",
        "Votants": "votants",
        "Exprimés": "exprimes",
    })
    df.columns = _normalize_col_names(df.columns)
    df["code_geo"] = df.apply(
        lambda r: normalize_codgeo(r.get("code_dep"), r.get("code_com")),
        axis=1
    )
    base_cols = ["code_geo", "code_dep", "libelle_dep", "libelle_commune", "inscrits", "votants", "exprimes"]
    available_base = [c for c in base_cols if c in df.columns]
    silver = df[available_base].copy()
    for col_name, series in _extract_voix_columns_wide(df, df.iloc[0], max_candidates=30).items():
        silver[col_name] = series
    return silver


def _read_elections_xlsx_2022_long(path) -> pd.DataFrame:
    """Lit le XLSX 2022 format long (une ligne par candidat par commune), retourne format wide."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name=0)
    df.columns = _normalize_col_names(df.columns)
    # Renommer colonnes 2022 (noms déjà normalisés sans accents)
    rename_2022 = {
        "Code du departement": "code_dep",
        "Libelle du departement": "libelle_dep",
        "Code de la commune": "code_com",
        "Libelle de la commune": "libelle_commune",
        "Inscrits": "inscrits",
        "Votants": "votants",
        "Exprimes": "exprimes",
    }
    df = df.rename(columns={k: v for k, v in rename_2022.items() if k in df.columns})
    if "code_dep" not in df.columns or "Voix" not in df.columns:
        return pd.DataFrame()
    df["code_geo"] = df.apply(
        lambda r: normalize_codgeo(r.get("code_dep"), r.get("code_com")),
        axis=1
    )
    # 2022 = format long : une ligne par (commune, candidat) -> on agrège puis on pivot
    agg_dict = {"inscrits": "first", "votants": "first", "exprimes": "first"}
    if "libelle_commune" in df.columns:
        agg_dict["libelle_commune"] = "first"
    if "libelle_dep" in df.columns:
        agg_dict["libelle_dep"] = "first"
    base = df.groupby("code_geo").agg(agg_dict).reset_index()
    if "code_dep" in df.columns:
        base["code_dep"] = df.groupby("code_geo")["code_dep"].first().values
    # Pivot : voix par candidat
    df["candidat"] = df["Nom"].astype(str).str.strip()
    pivot = df.pivot_table(index="code_geo", columns="candidat", values="Voix", aggfunc="sum").reset_index()
    new_pivot_cols = ["code_geo"]
    for c in pivot.columns:
        if c == "code_geo":
            continue
        safe = re.sub(r"[^a-zA-Z0-9]", "_", str(c))[:50]
        new_pivot_cols.append(f"voix_{safe}")
    pivot.columns = new_pivot_cols
    voix_cols = [c for c in pivot.columns if c.startswith("voix_")]
    silver = base.merge(pivot[["code_geo"] + voix_cols], on="code_geo", how="left")
    return silver


def process_elections_2012_presidentielle() -> pd.DataFrame:
    """Lit présidentielle 2012, retourne Silver."""
    entries = getattr(config, "BRONZE_ELECTIONS_EXTRA", [])
    path = next((e["path"] for e in entries if e.get("year") == 2012 and e.get("type") == "presidentielle"), None)
    return _read_elections_xls_2012(path)


def process_elections_2012_legislative() -> pd.DataFrame:
    """Lit législatives 2012, retourne Silver."""
    entries = getattr(config, "BRONZE_ELECTIONS_EXTRA", [])
    path = next((e["path"] for e in entries if e.get("year") == 2012 and e.get("type") == "legislative"), None)
    return _read_elections_xls_2012(path)


def process_elections_2022_legislative() -> pd.DataFrame:
    """Lit législatives 2022 (format long, subcom), retourne Silver."""
    entries = getattr(config, "BRONZE_ELECTIONS_EXTRA", [])
    path = next((e["path"] for e in entries if e.get("year") == 2022 and e.get("type") == "legislative"), None)
    return _read_elections_xlsx_2022_long(path)


# -----------------------------------------------------------------------------
# Géographie : population + indicateurs délinquance par commune et année
# -----------------------------------------------------------------------------

def process_geography() -> pd.DataFrame:
    """
    Lit le CSV géographie/délinquance, filtre par année (2016, 2017),
    agrège par commune : population + indicateurs (taux ou nombre).
    """
    if not config.BRONZE_GEOGRAPHY_CSV.exists():
        print(f"Fichier introuvable: {config.BRONZE_GEOGRAPHY_CSV}")
        return pd.DataFrame()

    # Lecture par chunks pour gros fichier
    chunks = []
    for chunk in pd.read_csv(
        config.BRONZE_GEOGRAPHY_CSV,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        chunksize=100_000,
        decimal=",",
        na_values=["NA", "na", ""],
    ):
        chunk = chunk[chunk["annee"].astype(str).str.isnumeric()]
        chunk["annee"] = pd.to_numeric(chunk["annee"], errors="coerce")
        chunk = chunk[chunk["annee"].isin(config.YEARS_FOR_ELECTION)]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Colonne code commune
    df["code_geo"] = df["CODGEO_2025"].astype(str).str.strip().str.replace('"', "")
    df["insee_pop"] = pd.to_numeric(df["insee_pop"], errors="coerce")

    # Une ligne par (code_geo, annee) avec population max (une seule valeur par commune/année)
    pop = df.groupby(["code_geo", "annee"])["insee_pop"].max().reset_index()
    pop = pop.rename(columns={"insee_pop": "population"})

    # Pivot : une colonne par indicateur (taux pour mille), une ligne par (code_geo, annee)
    df["indicateur"] = df["indicateur"].astype(str).str.strip().str.replace('"', "")
    df["taux_pour_mille"] = pd.to_numeric(df["taux_pour_mille"], errors="coerce")
    # Prendre le taux quand disponible, sinon complement_info_taux
    df["taux_val"] = df["taux_pour_mille"].fillna(pd.to_numeric(df.get("complement_info_taux", 0), errors="coerce"))

    pivot = df.pivot_table(
        index=["code_geo", "annee"],
        columns="indicateur",
        values="taux_val",
        aggfunc="first",
    ).reset_index()

    silver_geo = pop.merge(pivot, on=["code_geo", "annee"], how="left")
    return silver_geo


# -----------------------------------------------------------------------------
# État civil : mariages / décès / naissances (agrégés par département ou commune)
# -----------------------------------------------------------------------------

def process_etatcivil_mar() -> pd.DataFrame:
    """Lit le CSV des mariages 2017, agrège par département (DEPMAR)."""
    if not config.BRONZE_ETATCIVIL_MAR_CSV.exists():
        print(f"Fichier introuvable: {config.BRONZE_ETATCIVIL_MAR_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(config.BRONZE_ETATCIVIL_MAR_CSV, sep=";", encoding="utf-8", low_memory=False)
    df["DEPMAR"] = df["DEPMAR"].astype(str).str.zfill(2)
    # 2017 uniquement (AMAR = année mariage)
    df = df[df["AMAR"] == config.ELECTION_YEAR]
    agg = df.groupby("DEPMAR").size().reset_index(name="nb_mariages")
    agg = agg.rename(columns={"DEPMAR": "code_dep"})
    return agg


def process_etatcivil_dec() -> pd.DataFrame:
    """Lit le DBF des deces 2017, agrege par departement (depdec)."""
    if not getattr(config, "BRONZE_ETATCIVIL_DEC_DBF", None) or not config.BRONZE_ETATCIVIL_DEC_DBF.exists():
        return pd.DataFrame()
    try:
        from dbfread import DBF
    except ImportError:
        return pd.DataFrame()
    from collections import Counter
    cnt = Counter()
    for rec in DBF(str(config.BRONZE_ETATCIVIL_DEC_DBF)):
        dep = rec.get("depdec", "").strip().zfill(2)
        if dep:
            cnt[dep] += 1
    if not cnt:
        return pd.DataFrame()
    return pd.DataFrame([{"code_dep": k, "nb_deces": v} for k, v in sorted(cnt.items())])


def process_etatcivil_nais() -> pd.DataFrame:
    """Lit le DBF des naissances 2017, agrege par departement (depnais)."""
    if not getattr(config, "BRONZE_ETATCIVIL_NAIS_DBF", None) or not config.BRONZE_ETATCIVIL_NAIS_DBF.exists():
        return pd.DataFrame()
    try:
        from dbfread import DBF
    except ImportError:
        return pd.DataFrame()
    from collections import Counter
    cnt = Counter()
    for rec in DBF(str(config.BRONZE_ETATCIVIL_NAIS_DBF)):
        dep = rec.get("depnais", "").strip().zfill(2)
        if dep:
            cnt[dep] += 1
    if not cnt:
        return pd.DataFrame()
    return pd.DataFrame([{"code_dep": k, "nb_naissances": v} for k, v in sorted(cnt.items())])


# -----------------------------------------------------------------------------
# Emploi / chômage : enquête emploi (DBF/CSV) par année, puis chômage par dép
# -----------------------------------------------------------------------------

def _aggregate_emploi_dbf(dbf_path, year):
    """Parcourt le DBF emploi et retourne un dict (annee, nfrred) -> {actifs, chomeurs}."""
    from collections import defaultdict
    try:
        from dbfread import DBF
    except ImportError:
        return {}
    stats = defaultdict(lambda: {"actifs": 0, "chomeurs": 0})
    year_str = str(year)
    for rec in DBF(str(dbf_path)):
        if rec.get("ANNEE", "").strip() != year_str:
            continue
        acteu = rec.get("ACTEU", "").strip()
        if acteu not in ("1", "2"):
            continue
        key = (year_str, rec.get("NFRRED", "").strip() or "0")
        stats[key]["actifs"] += 1
        if acteu == "2":
            stats[key]["chomeurs"] += 1
    return dict(stats)


def _aggregate_emploi_csv(csv_path, year):
    """Parcourt le CSV enquête emploi et retourne un dict (annee, region) -> {actifs, chomeurs}."""
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", low_memory=False)
    except Exception:
        return {}
    year_str = str(year)
    df = df[df["ANNEE"].astype(str).str.strip() == year_str]
    df = df[df["ACTEU"].astype(str).str.strip().isin(("1", "2"))]
    if df.empty:
        return {}
    col_region = "NFRRED" if "NFRRED" in df.columns else "ENFRED"
    if col_region not in df.columns:
        return {}
    df["_region"] = df[col_region].astype(str).str.strip().fillna("0")
    df["_chomeur"] = (df["ACTEU"].astype(str).str.strip() == "2").astype(int)
    agg = df.groupby("_region").agg(actifs=("_chomeur", "count"), chomeurs=("_chomeur", "sum")).reset_index()
    stats = {(year_str, row["_region"] or "0"): {"actifs": row["actifs"], "chomeurs": row["chomeurs"]} for _, row in agg.iterrows()}
    return stats


def _emploi_stats_to_rows(stats):
    """Convertit stats (annee, nfrred) -> {actifs, chomeurs} en liste de lignes pour DataFrame."""
    rows = []
    for (annee, nfrred), v in sorted(stats.items()):
        taux = round(100 * v["chomeurs"] / v["actifs"], 4) if v["actifs"] > 0 else 0.0
        rows.append({"annee": annee, "code_region": nfrred or "0", "taux_chomage": taux})
    return rows


def _aggregate_emploi_file(path, year):
    """Agrège un fichier emploi (DBF ou CSV) pour une année ; retourne dict stats ou vide."""
    if not path or not Path(path).exists():
        return {}
    if str(path).lower().endswith(".csv"):
        return _aggregate_emploi_csv(path, year)
    return _aggregate_emploi_dbf(path, year)


def process_emploi() -> pd.DataFrame:
    """Lit l'enquête emploi pour l'année d'élection (compatibilité)."""
    return process_emploi_for_year(config.ELECTION_YEAR)


def process_emploi_for_year(year: int) -> pd.DataFrame:
    """
    Lit l'enquête emploi (DBF ou CSV) pour une année donnée.
    ACTEU: 1=occupe, 2=chomeur, 3=inactif. Taux = chomeurs / actifs par région.
    """
    by_year = getattr(config, "BRONZE_EMPLOI_BY_YEAR", None)
    path = by_year.get(year) if by_year else None
    if not path and year == config.ELECTION_YEAR:
        path = getattr(config, "BRONZE_EMPLOI_DBF", None)
    stats = _aggregate_emploi_file(path, year) if path else {}
    if not stats:
        return pd.DataFrame()
    return pd.DataFrame(_emploi_stats_to_rows(stats))


def process_deces_communes() -> pd.DataFrame:
    """Lit le CSV deces par commune (data.gouv), filtre COM + annee election, retourne code_geo, nb_deces."""
    path = getattr(config, "BRONZE_DECES_COMMUNES_CSV", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    df = df[(df["GEO_OBJECT"] == "COM") & (df["TIME_PERIOD"] == config.ELECTION_YEAR) & (df["OBS_STATUS"] == "A")]
    df["code_geo"] = df["GEO"].astype(str).str.strip().str.zfill(5)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce").fillna(0).astype(int)
    return df.groupby("code_geo", as_index=False)["OBS_VALUE"].sum().rename(columns={"OBS_VALUE": "nb_deces"})


def process_nais_communes() -> pd.DataFrame:
    """Lit le CSV naissances par commune (data.gouv), filtre COM + annee election, retourne code_geo, nb_naissances."""
    path = getattr(config, "BRONZE_NAIS_COMMUNES_CSV", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    df = df[(df["GEO_OBJECT"] == "COM") & (df["TIME_PERIOD"] == config.ELECTION_YEAR) & (df["OBS_STATUS"] == "A")]
    df["code_geo"] = df["GEO"].astype(str).str.strip().str.zfill(5)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce").fillna(0).astype(int)
    return df.groupby("code_geo", as_index=False)["OBS_VALUE"].sum().rename(columns={"OBS_VALUE": "nb_naissances"})


# -----------------------------------------------------------------------------
# Autres sources Silver : circo, population dép, Filosofi, diplômes, pop commune
# -----------------------------------------------------------------------------

def process_circo() -> pd.DataFrame:
    """Lit circo_composition.xlsx : commune -> circonscription et region (code_geo, circo, REG)."""
    path = getattr(config, "BRONZE_CIRCO_XLSX", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="table")
    df["code_geo"] = df["COMMUNE_RESID"].astype(str).str.strip().str.zfill(5)
    return df[["code_geo", "circo", "REG"]].drop_duplicates("code_geo")


def process_population_dep() -> pd.DataFrame:
    """Lit DS_ESTIMATION_POPULATION : population au 1er janv par departement (annee election)."""
    path = getattr(config, "BRONZE_POPULATION_CSV", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    df = df[
        (df["GEO_OBJECT"] == "DEP")
        & (df["EP_MEASURE"] == "POP_JAN_1ST")
        & (df["TIME_PERIOD"] == config.ELECTION_YEAR)
        & (df["SEX"] == "_T")
    ]
    df["code_dep"] = df["GEO"].astype(str).str.strip().str.zfill(2)
    df["pop_dep"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return df[["code_dep", "pop_dep"]].dropna(subset=["pop_dep"]).drop_duplicates("code_dep")


def process_filosofi_com() -> pd.DataFrame:
    """Lit Filosofi par commune (revenus, pauvreté). Retourne code_geo, med_niveau_vie, taux_pauvrette."""
    path = getattr(config, "BRONZE_FILOSOFI_COM_CSV", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "CODGEO" not in df.columns or "MED17" not in df.columns:
        return pd.DataFrame()
    df["code_geo"] = df["CODGEO"].astype(str).str.strip().str.zfill(5)
    df["med_niveau_vie"] = pd.to_numeric(df["MED17"], errors="coerce")
    df["taux_pauvrette"] = pd.to_numeric(df.get("TP6017", pd.Series(dtype=float)), errors="coerce")
    out = df[["code_geo", "med_niveau_vie", "taux_pauvrette"]].copy()
    out = out[out["code_geo"].str.match(r"^\d{5}$|^2[AB]\d{3}$", na=False)]
    return out.dropna(subset=["code_geo"]).drop_duplicates("code_geo")


def process_diplomes() -> pd.DataFrame:
    """Lit recensement diplômes (DS_RP_DIPLOMES), agrège par commune : part_sans_diplome, part_bac_plus."""
    path = getattr(config, "BRONZE_DIPLOMES_CSV", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False,
                     quoting=1, on_bad_lines="skip")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    for need in ["GEO", "EDUC", "OBS_VALUE", "GEO_OBJECT", "TIME_PERIOD"]:
        if need not in df.columns:
            return pd.DataFrame()
    df = df[(df["GEO_OBJECT"].astype(str).str.strip() == "COM")].copy()
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    # Prendre 2022 si dispo, sinon la plus récente année disponible
    years = sorted(df["TIME_PERIOD"].dropna().unique(), reverse=True)
    if 2022 in years:
        year = 2022
    elif years:
        year = int(years[0])
    else:
        year = None
    if year is None:
        return pd.DataFrame()
    df = df[df["TIME_PERIOD"] == year]
    df["code_geo"] = df["GEO"].astype(str).str.strip().str.zfill(5)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce").fillna(0)
    df["EDUC"] = df["EDUC"].astype(str).str.strip()
    # _T = total 15+ ; 001T003 = sans diplôme ; 500T702_RP/600/700 = bac et plus
    total = df[df["EDUC"] == "_T"].groupby("code_geo")["OBS_VALUE"].sum()
    sans = df[df["EDUC"].str.contains("001T003", na=False)].groupby("code_geo")["OBS_VALUE"].sum()
    sup = df[df["EDUC"].isin(["500T702_RP", "600_RP", "700_RP"])].groupby("code_geo")["OBS_VALUE"].sum()
    out = pd.DataFrame({"code_geo": total.index, "pop_15_plus": total.values})
    out = out.merge(sans.rename("pop_sans_diplome"), left_on="code_geo", right_index=True, how="left", validate="1:1")
    out = out.merge(sup.rename("pop_bac_plus"), left_on="code_geo", right_index=True, how="left", validate="1:1")
    out["pop_sans_diplome"] = out["pop_sans_diplome"].fillna(0)
    out["pop_bac_plus"] = out["pop_bac_plus"].fillna(0)
    out["part_sans_diplome"] = (out["pop_sans_diplome"] / out["pop_15_plus"].replace(0, np.nan)).round(4)
    out["part_bac_plus"] = (out["pop_bac_plus"] / out["pop_15_plus"].replace(0, np.nan)).round(4)
    return out[["code_geo", "part_sans_diplome", "part_bac_plus"]].drop_duplicates("code_geo")


def process_population_commune() -> pd.DataFrame:
    """Lit populations historiques : population par commune pour l'année d'élection (code_geo, pop_commune)."""
    path = getattr(config, "BRONZE_POP_HISTORIQUES_CSV", None)
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False,
                     quoting=1, on_bad_lines="skip")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    geo_col = "GEO" if "GEO" in df.columns else next((c for c in df.columns if "GEO" in c.upper()), None)
    period_col = next((c for c in df.columns if "TIME" in c.upper() or "PERIOD" in c.upper()), None)
    value_col = next((c for c in df.columns if "OBS" in c.upper() or "VALUE" in c.upper()), None)
    if not geo_col or not value_col:
        return pd.DataFrame()
    obj_col = next((c for c in df.columns if "OBJECT" in c.upper()), None)
    if obj_col:
        df = df[df[obj_col].astype(str).str.strip().str.upper() == "COM"]
    year = config.ELECTION_YEAR
    if period_col and period_col in df.columns:
        df[period_col] = pd.to_numeric(df[period_col], errors="coerce")
        df = df[df[period_col] == year]
    df["code_geo"] = df[geo_col].astype(str).str.strip().str.zfill(5)
    df["pop_commune"] = pd.to_numeric(df[value_col], errors="coerce")
    out = df.groupby("code_geo", as_index=False)["pop_commune"].sum()
    return out[out["code_geo"].str.match(r"^\d{5}$|^2[AB]\d{3}$", na=False)]


def process_chomage_dep() -> pd.DataFrame:
    """
    Lit famille_TAUX-CHOMAGE (Insee) : taux de chômage localisé par département.
    Combine caractéristiques.csv (idBank -> zone) et valeurs_trimestrielles.csv (taux par trimestre).
    Retourne code_dep, taux_chomage (moyenne des 4 trimestres de l'année d'élection).
    """
    car_path = getattr(config, "BRONZE_CHOMAGE_DEP_CARACT", None)
    val_path = getattr(config, "BRONZE_CHOMAGE_DEP_VALEURS", None)
    if not car_path or not car_path.exists() or not val_path or not val_path.exists():
        return pd.DataFrame()
    try:
        car = pd.read_csv(car_path, sep=";", encoding="utf-8")
        val = pd.read_csv(val_path, sep=";", encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    car.columns = [str(c).strip().strip("'\"") for c in car.columns]
    val.columns = [str(c).strip().strip("'\"") for c in val.columns]
    zone_col = [c for c in car.columns if "zone" in c.lower() or "geo" in c.lower()]
    if not zone_col:
        return pd.DataFrame()
    zone_col = zone_col[0]
    id_col = next((c for c in car.columns if "idbank" in c.lower()), "idBank")
    zone_str = car[zone_col].astype(str).str.strip()
    car = car[zone_str.str.match(r"^(\d{2}|2A|2B)\s*-\s", na=False)].copy()
    if car.empty:
        return pd.DataFrame()
    car["code_dep"] = car[zone_col].astype(str).str.strip().str[:2]
    car.loc[car[zone_col].astype(str).str.strip().str.upper().str.startswith("2A"), "code_dep"] = "2A"
    car.loc[car[zone_col].astype(str).str.strip().str.upper().str.startswith("2B"), "code_dep"] = "2B"
    # idBank peut être "001515842" (car) ou 1515842.0 (val) -> normaliser en int pour la jointure
    car["_id"] = pd.to_numeric(car[id_col], errors="coerce").astype("Int64")
    id_to_dep = car.dropna(subset=["_id"]).set_index("_id")["code_dep"].to_dict()
    val = val[val["Libellé"].astype(str).str.strip() != "Codes"].copy()
    year = config.ELECTION_YEAR
    # Moyenne des 4 trimestres de l'année pour avoir un taux annuel par département
    trim_cols = [f"{year}-T1", f"{year}-T2", f"{year}-T3", f"{year}-T4"]
    trim_cols = [c for c in trim_cols if c in val.columns]
    if not trim_cols:
        return pd.DataFrame()
    val["_id"] = pd.to_numeric(val[id_col], errors="coerce")
    val["code_dep"] = val["_id"].map(id_to_dep)
    val = val[val["code_dep"].notna()]
    if val.empty:
        return pd.DataFrame()
    val["taux_chomage"] = val[trim_cols].apply(
        lambda row: pd.to_numeric(row, errors="coerce").mean(), axis=1
    )
    out = val.groupby("code_dep", as_index=False)["taux_chomage"].mean()
    out["taux_chomage"] = out["taux_chomage"].round(4)
    return out


# -----------------------------------------------------------------------------
# Écriture Silver et orchestration (run)
# -----------------------------------------------------------------------------

def _write_silver_if_not_empty(df, filename):
    """Ecrit le DataFrame en CSV dans SILVER_DIR et affiche un message si non vide."""
    if df.empty:
        return
    path = config.SILVER_DIR / filename
    df.to_csv(path, index=False, sep=";", encoding="utf-8")
    print(f"  -> {path} ({len(df)} lignes)")


def _silver_tasks(year):
    """Liste des étapes (label, fonction, nom fichier Silver) exécutées dans l'ordre."""
    tasks = [
        ("elections", process_elections, f"elections_{year}.csv"),
        ("elections 2012 presidentielle", process_elections_2012_presidentielle, "elections_2012_presidentielle.csv"),
        ("elections 2012 legislative", process_elections_2012_legislative, "elections_2012_legislative.csv"),
        ("elections 2022 legislative", process_elections_2022_legislative, "elections_2022_legislative.csv"),
        ("geographie / delinquance", process_geography, f"geographie_{year}.csv"),
        ("etat civil (mariages)", process_etatcivil_mar, f"etatcivil_mar_{year}.csv"),
        ("etat civil (deces, dBase)", process_etatcivil_dec, f"etatcivil_dec_{year}.csv"),
        ("etat civil (naissances, dBase)", process_etatcivil_nais, f"etatcivil_nais_{year}.csv"),
        ("deces par commune (CSV)", process_deces_communes, f"deces_communes_{year}.csv"),
        ("naissances par commune (CSV)", process_nais_communes, f"nais_communes_{year}.csv"),
    ]
    by_year = getattr(config, "BRONZE_EMPLOI_BY_YEAR", None)
    if by_year:
        for ey in sorted(by_year.keys()):
            if by_year[ey] and Path(by_year[ey]).exists():
                tasks.append((f"emploi / chomage {ey}", lambda y=ey: process_emploi_for_year(y), f"emploi_{ey}.csv"))
    tasks.append(("chomage par departement (Insee)", process_chomage_dep, f"chomage_dep_{year}.csv"))
    tasks.extend([
        ("circonscriptions (circo, region)", process_circo, f"circo_{year}.csv"),
        ("population par departement", process_population_dep, f"population_dep_{year}.csv"),
        ("Filosofi revenus/pauvrete commune", process_filosofi_com, "filosofi_com_2017.csv"),
        ("diplomes recensement", process_diplomes, "diplomes_2022.csv"),
        ("population communale (historiques)", process_population_commune, f"population_commune_{year}.csv"),
    ])
    return tasks


def run():
    ensure_silver_dir()
    y = config.ELECTION_YEAR
    for label, process_fn, filename in _silver_tasks(y):
        print(f"Bronze -> Silver : {label}...")
        _write_silver_if_not_empty(process_fn(), filename)
    print("Silver pret.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=None, help="Annee election (defaut: config.ELECTION_YEAR)")
    args = p.parse_args()
    if args.year is not None:
        config.ELECTION_YEAR = args.year
        config.YEARS_FOR_ELECTION = [args.year - 1, args.year]
    run()
