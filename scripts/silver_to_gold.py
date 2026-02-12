# -*- coding: utf-8 -*-
"""
Pipeline Silver -> Gold : jointure des donnees Silver, filtre communes > MIN_POPULATION,
remplissage des tables du schema MCD (geo, vote, DateDuVote, stat, candidat,
parti_politique, situe, possede) + table resultat (voix par candidat par commune).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def ensure_gold_dir():
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)


def _load_extra_elections(communes_10k_set, exclude_years=None):
    """
    Charge les elections supplementaires (2012 pres/leg uniquement par defaut).
    Retourne un DataFrame par fichier avec code_geo et colonnes part_voix_<CANDIDAT>_<year>_<type>.
    exclude_years : annees a ne pas charger (defaut : [2022], trop de NaN par commune).
    Limite : top 15 candidats par election (par total voix).
    """
    extra = getattr(config, "BRONZE_ELECTIONS_EXTRA", [])
    if not extra:
        return []
    exclude_years = exclude_years if exclude_years is not None else [2022]
    results = []
    for entry in extra:
        year, typ = entry.get("year"), entry.get("type")
        if year in exclude_years:
            continue
        path = config.SILVER_DIR / f"elections_{year}_{typ}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
        if df.empty:
            continue
        df["code_geo"] = df["code_geo"].astype(str).str.strip().str.zfill(5)
        df = df[df["code_geo"].isin(communes_10k_set)]
        if df.empty:
            continue
        voix_cols = [c for c in df.columns if c.startswith("voix_")]
        if not voix_cols:
            continue
        df["_exprimes"] = df[voix_cols].sum(axis=1)
        df = df[df["_exprimes"] > 0]
        # Garder top 15 candidats par total voix
        totals = df[voix_cols].sum().sort_values(ascending=False)
        keep = totals.head(15).index.tolist()
        part_df = df[["code_geo"]].copy()
        for col in keep:
            candidat = col.replace("voix_", "")
            part_df[f"part_voix_{candidat}_{year}_{typ}"] = (
                df[col] / df["_exprimes"]
            ).round(6)
        results.append(part_df)
    return results


def load_silver():
    """Charge les DataFrames Silver."""
    elections = pd.read_csv(
        config.SILVER_DIR / f"elections_{config.ELECTION_YEAR}.csv",
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )
    geography = pd.read_csv(
        config.SILVER_DIR / f"geographie_{config.ELECTION_YEAR}.csv",
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )
    etatcivil = pd.read_csv(
        config.SILVER_DIR / f"etatcivil_mar_{config.ELECTION_YEAR}.csv",
        sep=";",
        encoding="utf-8",
    )
    etatcivil_dec = None
    etatcivil_nais = None
    deces_communes = None
    nais_communes = None
    emploi = None
    dec_path = config.SILVER_DIR / f"etatcivil_dec_{config.ELECTION_YEAR}.csv"
    nais_path = config.SILVER_DIR / f"etatcivil_nais_{config.ELECTION_YEAR}.csv"
    dec_com_path = config.SILVER_DIR / f"deces_communes_{config.ELECTION_YEAR}.csv"
    nais_com_path = config.SILVER_DIR / f"nais_communes_{config.ELECTION_YEAR}.csv"
    emploi_path = config.SILVER_DIR / f"emploi_{config.ELECTION_YEAR}.csv"
    if dec_path.exists():
        etatcivil_dec = pd.read_csv(dec_path, sep=";", encoding="utf-8")
    if nais_path.exists():
        etatcivil_nais = pd.read_csv(nais_path, sep=";", encoding="utf-8")
    if dec_com_path.exists():
        deces_communes = pd.read_csv(dec_com_path, sep=";", encoding="utf-8")
    if nais_com_path.exists():
        nais_communes = pd.read_csv(nais_com_path, sep=";", encoding="utf-8")
    if emploi_path.exists():
        emploi = pd.read_csv(emploi_path, sep=";", encoding="utf-8")
    chomage_dep = None
    chomage_dep_path = config.SILVER_DIR / f"chomage_dep_{config.ELECTION_YEAR}.csv"
    if chomage_dep_path.exists():
        chomage_dep = pd.read_csv(chomage_dep_path, sep=";", encoding="utf-8")
    circo = None
    population_dep = None
    circo_path = config.SILVER_DIR / f"circo_{config.ELECTION_YEAR}.csv"
    pop_dep_path = config.SILVER_DIR / f"population_dep_{config.ELECTION_YEAR}.csv"
    if circo_path.exists():
        circo = pd.read_csv(circo_path, sep=";", encoding="utf-8")
    if pop_dep_path.exists():
        population_dep = pd.read_csv(pop_dep_path, sep=";", encoding="utf-8")
    filosofi_com = None
    diplomes = None
    population_commune = None
    if (config.SILVER_DIR / "filosofi_com_2017.csv").exists():
        filosofi_com = pd.read_csv(config.SILVER_DIR / "filosofi_com_2017.csv", sep=";", encoding="utf-8")
    if (config.SILVER_DIR / "diplomes_2022.csv").exists():
        diplomes = pd.read_csv(config.SILVER_DIR / "diplomes_2022.csv", sep=";", encoding="utf-8")
    pop_com_path = config.SILVER_DIR / f"population_commune_{config.ELECTION_YEAR}.csv"
    if pop_com_path.exists():
        population_commune = pd.read_csv(pop_com_path, sep=";", encoding="utf-8")
    return (elections, geography, etatcivil, etatcivil_dec, etatcivil_nais,
            deces_communes, nais_communes, emploi, chomage_dep, circo, population_dep,
            filosofi_com, diplomes, population_commune)


def _filter_communes_and_build_geo_tables(geography, elections):
    """Filtre communes >= MIN_POPULATION, construit geo_to_id, geo_df, elec ; ecrit geo, vote, DateDuVote."""
    pop = geography.groupby("code_geo")["population"].max().reset_index()
    communes_10k = set(
        pop[pop["population"] >= config.MIN_POPULATION]["code_geo"].astype(str).tolist()
    )
    elections = elections.copy()
    elections["code_geo"] = elections["code_geo"].astype(str).str.strip()
    elec = elections[elections["code_geo"].isin(communes_10k)].copy()
    code_geo_list = sorted(elec["code_geo"].unique())
    geo_to_id = {c: i + 1 for i, c in enumerate(code_geo_list)}
    elec["Id_geo"] = elec["code_geo"].map(geo_to_id)
    geo_df = elec[["code_geo", "libelle_commune", "libelle_dep"]].drop_duplicates("code_geo")
    geo_df = geo_df.sort_values("code_geo")
    geo_df["Id_geo"] = geo_df["code_geo"].map(geo_to_id)
    geo_df = geo_df[["Id_geo", "code_geo", "libelle_commune", "libelle_dep"]]
    geo_df = geo_df.rename(columns={"libelle_commune": "ville", "libelle_dep": "departement"})
    geo_df.to_csv(config.GOLD_DIR / "geo.csv", index=False, sep=";", encoding="utf-8")
    pd.DataFrame([{"Id_vote": 1}]).to_csv(config.GOLD_DIR / "vote.csv", index=False, sep=";", encoding="utf-8")
    pd.DataFrame([{"Id_DateDuVote": 1, "fuseauxHoraire": "Europe/Paris", "Id_vote": 1}]).to_csv(
        config.GOLD_DIR / "DateDuVote.csv", index=False, sep=";", encoding="utf-8"
    )
    return pop, communes_10k, elec, geo_to_id, geo_df


def _build_candidats_and_partis(elec):
    """Construit et ecrit candidat, parti_politique ; retourne voix_cols et nom_to_id_candidat."""
    voix_cols = [c for c in elec.columns if c.startswith("voix_")]
    candidat_noms = [c.replace("voix_", "").replace("_", " ") for c in voix_cols]
    candidat_df = pd.DataFrame({
        "Id_candidat": range(1, len(candidat_noms) + 1),
        "nom": [n.split()[-1] if n.split() else n for n in candidat_noms],
        "prenom": [" ".join(n.split()[:-1]) if len(n.split()) > 1 else "" for n in candidat_noms],
        "Id_vote": 1,
    })
    candidat_df.to_csv(config.GOLD_DIR / "candidat.csv", index=False, sep=";", encoding="utf-8")
    parti_df = pd.DataFrame({
        "Id_parti_politique": range(1, len(candidat_noms) + 1),
        "nom": ["Parti"] * len(candidat_noms),
        "orientation": "N/A",
        "Id_candidat": range(1, len(candidat_noms) + 1),
    })
    parti_df.to_csv(config.GOLD_DIR / "parti_politique.csv", index=False, sep=";", encoding="utf-8")
    nom_to_id_candidat = {vc.replace("voix_", ""): i + 1 for i, vc in enumerate(voix_cols)}
    return voix_cols, nom_to_id_candidat


def _stat_apply_deces(geo_2017, deces_communes, etatcivil_dec):
    """Remplit TauxDec (priorité commune CSV, sinon département DBF)."""
    if deces_communes is not None and not deces_communes.empty:
        dec = deces_communes.copy()
        dec["code_geo"] = dec["code_geo"].astype(str).str.zfill(5)
        geo_2017 = geo_2017.merge(dec[["code_geo", "nb_deces"]], on="code_geo", how="left")
        geo_2017["nb_deces"] = geo_2017["nb_deces"].fillna(0)
        geo_2017["TauxDec"] = (
            geo_2017["nb_deces"] / geo_2017["population"].replace(0, np.nan) * 1000
        ).round(4).astype(str)
        return geo_2017
    if etatcivil_dec is not None and not etatcivil_dec.empty:
        dep_dec = etatcivil_dec.set_index("code_dep")["nb_deces"].to_dict()
        geo_2017["nb_deces_dep"] = geo_2017["code_dep"].map(dep_dec)
        geo_2017["TauxDec"] = (
            geo_2017["nb_deces_dep"] / geo_2017["pop_dep"].replace(0, np.nan) * 1000
        ).round(4).astype(str)
    return geo_2017


def _stat_apply_nais(geo_2017, nais_communes, etatcivil_nais):
    """Remplit tauxNatalite (priorité commune CSV, sinon département DBF)."""
    if nais_communes is not None and not nais_communes.empty:
        nais = nais_communes.copy()
        nais["code_geo"] = nais["code_geo"].astype(str).str.zfill(5)
        geo_2017 = geo_2017.merge(nais[["code_geo", "nb_naissances"]], on="code_geo", how="left")
        geo_2017["nb_naissances"] = geo_2017["nb_naissances"].fillna(0)
        geo_2017["tauxNatalite"] = (
            geo_2017["nb_naissances"] / geo_2017["population"].replace(0, np.nan) * 1000
        ).round(4).astype(str)
        return geo_2017
    if etatcivil_nais is not None and not etatcivil_nais.empty:
        dep_nais = etatcivil_nais.set_index("code_dep")["nb_naissances"].to_dict()
        geo_2017["nb_nais_dep"] = geo_2017["code_dep"].map(dep_nais)
        geo_2017["tauxNatalite"] = (
            geo_2017["nb_nais_dep"] / geo_2017["pop_dep"].replace(0, np.nan) * 1000
        ).round(4).astype(str)
    return geo_2017


def _stat_merge_optional(geo_2017, opt_sources):
    """Fusionne Filosofi, diplômes, population communale si présents."""
    for key, cols in [
        ("filosofi_com", ["code_geo", "med_niveau_vie", "taux_pauvrette"]),
        ("diplomes", ["code_geo", "part_sans_diplome", "part_bac_plus"]),
        ("population_commune", ["code_geo", "pop_commune"]),
    ]:
        df = opt_sources.get(key)
        if df is not None and not df.empty:
            df = df.copy()
            df["code_geo"] = df["code_geo"].astype(str).str.strip().str.zfill(5)
            geo_2017 = geo_2017.merge(df[cols], on="code_geo", how="left")
    return geo_2017


def _stat_fill_chomage(stat_per_geo, chomage_dep, emploi):
    """Remplit tauxChomage (département ou moyenne emploi)."""
    if (chomage_dep is not None and not chomage_dep.empty
            and "code_dep" in chomage_dep.columns and "taux_chomage" in chomage_dep.columns):
        chomage_dep = chomage_dep.copy()
        chomage_dep["code_dep"] = chomage_dep["code_dep"].astype(str).str.strip().str.zfill(2)
        dep_to_taux = chomage_dep.set_index("code_dep")["taux_chomage"].to_dict()
        stat_per_geo["tauxChomage"] = stat_per_geo["code_dep"].map(
            lambda d: dep_to_taux.get(str(d).zfill(2), np.nan)
        )
        stat_per_geo["tauxChomage"] = stat_per_geo["tauxChomage"].fillna("N/A").astype(str)
        return stat_per_geo
    taux = "N/A"
    if emploi is not None and not emploi.empty and "taux_chomage" in emploi.columns:
        taux = str(round(emploi["taux_chomage"].mean(), 4))
    stat_per_geo["tauxChomage"] = taux
    return stat_per_geo


def _build_stat_table(geography, communes_10k, geo_to_id, pop, elec, etatcivil, opt_sources=None):
    """Construit la table stat par commune. opt_sources: dict avec etatcivil_dec, etatcivil_nais,
    deces_communes, nais_communes, emploi, chomage_dep, filosofi_com, diplomes, population_commune."""
    opt = opt_sources or {}
    geo_2017 = geography[geography["annee"] == config.ELECTION_YEAR].copy()
    geo_2017["code_geo"] = geo_2017["code_geo"].astype(str)
    geo_2017 = geo_2017[geo_2017["code_geo"].isin(communes_10k)]
    ind_cols = [c for c in geo_2017.columns
                if c not in ("code_geo", "annee", "population")
                and pd.api.types.is_numeric_dtype(geo_2017[c])]
    geo_2017["tauxSec"] = geo_2017[ind_cols].mean(axis=1).round(4).astype(str) if ind_cols else "0"
    elec["code_dep"] = elec["code_dep"].astype(str).str.zfill(2)
    dep_mar = etatcivil.set_index("code_dep")["nb_mariages"].to_dict()
    geo_2017["code_dep"] = geo_2017["code_geo"].str[:2]
    geo_2017["TauxMariage"] = geo_2017["code_dep"].map(lambda d: str(dep_mar.get(d, 0)))
    pop_dep = pop.copy()
    pop_dep["code_dep"] = pop_dep["code_geo"].astype(str).str[:2]
    pop_dep = pop_dep.groupby("code_dep")["population"].sum().to_dict()
    geo_2017["pop_dep"] = geo_2017["code_dep"].map(pop_dep)
    geo_2017["TauxDec"] = "N/A"
    geo_2017["tauxNatalite"] = "N/A"
    geo_2017 = _stat_apply_deces(geo_2017, opt.get("deces_communes"), opt.get("etatcivil_dec"))
    geo_2017 = _stat_apply_nais(geo_2017, opt.get("nais_communes"), opt.get("etatcivil_nais"))
    geo_2017 = _stat_merge_optional(geo_2017, opt)
    agg_dict = {"tauxSec": "first", "TauxMariage": "first", "TauxDec": "first", "tauxNatalite": "first", "code_dep": "first"}
    for col in ["med_niveau_vie", "taux_pauvrette", "part_sans_diplome", "part_bac_plus", "pop_commune"]:
        if col in geo_2017.columns:
            agg_dict[col] = "first"
    stat_per_geo = geo_2017.groupby("code_geo").agg(agg_dict).reset_index()
    stat_per_geo["Id_geo"] = stat_per_geo["code_geo"].map(geo_to_id)
    stat_per_geo["Id_stat"] = stat_per_geo["Id_geo"]
    stat_per_geo = _stat_fill_chomage(stat_per_geo, opt.get("chomage_dep"), opt.get("emploi"))
    base_stat_cols = ["Id_stat", "tauxChomage", "tauxNatalite", "TauxDec", "TauxMariage", "tauxSec"]
    extra_stat_cols = [c for c in ["med_niveau_vie", "taux_pauvrette", "part_sans_diplome", "part_bac_plus", "pop_commune"]
                      if c in stat_per_geo.columns]
    stat_df = stat_per_geo[base_stat_cols + extra_stat_cols]
    stat_df.to_csv(config.GOLD_DIR / "stat.csv", index=False, sep=";", encoding="utf-8")
    geo_to_id_stat = stat_per_geo.set_index("Id_geo")["Id_stat"].to_dict()
    return stat_per_geo, geo_to_id_stat


def _build_resultat_rows(elec, voix_cols, nom_to_id_candidat):
    """Construit la liste des lignes pour la table resultat (Id_geo, Id_candidat, nb_voix)."""
    rows = []
    for _, row in elec.iterrows():
        id_geo = row["Id_geo"]
        for voix_col in voix_cols:
            id_candidat = nom_to_id_candidat.get(voix_col.replace("voix_", ""))
            if id_candidat is None:
                continue
            nb_voix = row.get(voix_col)
            rows.append({"Id_geo": id_geo, "Id_candidat": id_candidat, "nb_voix": int(nb_voix if pd.notna(nb_voix) else 0)})
    return rows


def _merge_extra_parts_into_vue(vue_ml, extra_elections_parts):
    """Fusionne les DataFrames part_voix (2012 pres/leg) dans vue_ml sur code_geo. 2022 exclu (trop de NaN)."""
    if not extra_elections_parts:
        return vue_ml
    for part_df in extra_elections_parts:
        if part_df is None or part_df.empty:
            continue
        part_cols = [c for c in part_df.columns if c != "code_geo"]
        if part_cols:
            vue_ml = vue_ml.merge(part_df[["code_geo"] + part_cols], on="code_geo", how="left")
    return vue_ml


def _apply_circo_and_pop(vue_ml, circo, population_dep):
    """Applique les merges circo et population_dep sur vue_ml."""
    if circo is not None and not circo.empty:
        circo = circo.copy()
        circo["code_geo"] = circo["code_geo"].astype(str).str.zfill(5)
        vue_ml = vue_ml.merge(circo[["code_geo", "circo", "REG"]], on="code_geo", how="left")
    if population_dep is not None and not population_dep.empty:
        vue_ml["code_dep"] = vue_ml["code_geo"].astype(str).str[:2]
        vue_ml = vue_ml.merge(population_dep, on="code_dep", how="left")
    return vue_ml


def _write_resultat_and_vue_ml(elec, geo_df, stat_per_geo, voix_cols, nom_to_id_candidat,
                               geo_to_id_stat, geo_to_id, circo=None, population_dep=None,
                               extra_elections_parts=None, output_view_basename=None):
    """Ecrit situe, possede, resultat et la vue ML. output_view_basename : nom du fichier vue (ex. gold_ml_view_2022.csv)."""
    situe_df = pd.DataFrame({"Id_geo": list(geo_to_id.values()), "Id_vote": 1})
    situe_df.to_csv(config.GOLD_DIR / "situe.csv", index=False, sep=";", encoding="utf-8")
    possede_df = pd.DataFrame({
        "Id_geo": list(geo_to_id.values()),
        "Id_stat": [geo_to_id_stat.get(i) for i in geo_to_id.values()],
    })
    possede_df.to_csv(config.GOLD_DIR / "possede.csv", index=False, sep=";", encoding="utf-8")
    pd.DataFrame(_build_resultat_rows(elec, voix_cols, nom_to_id_candidat)).to_csv(
        config.GOLD_DIR / "resultat.csv", index=False, sep=";", encoding="utf-8"
    )
    stat_cols = [c for c in ["Id_geo", "tauxChomage", "tauxSec", "TauxMariage", "TauxDec", "tauxNatalite",
                             "med_niveau_vie", "taux_pauvrette", "part_sans_diplome", "part_bac_plus", "pop_commune"] if c in stat_per_geo.columns]
    vue_ml = geo_df.merge(
        stat_per_geo[["Id_geo"] + [c for c in stat_cols if c != "Id_geo"]],
        on="Id_geo", how="left"
    )
    vue_ml = _apply_circo_and_pop(vue_ml, circo, population_dep)
    vue_ml = vue_ml.merge(elec[["Id_geo"] + voix_cols], on="Id_geo", how="left")
    # Parts des voix 2017 en % (comme 2012), puis on retire les voix_* pour eviter confusion et fuite
    exprimes = vue_ml[voix_cols].sum(axis=1).replace(0, np.nan)
    for col in voix_cols:
        cand = col.replace("voix_", "")
        vue_ml[f"part_voix_2017_{cand}"] = (vue_ml[col] / exprimes).round(6)
    vue_ml = vue_ml.drop(columns=voix_cols, errors="ignore")
    vue_ml = _merge_extra_parts_into_vue(vue_ml, extra_elections_parts or [])
    view_name = output_view_basename if output_view_basename else "gold_ml_view.csv"
    vue_ml.to_csv(config.GOLD_DIR / view_name, index=False, sep=";", encoding="utf-8")


def run(output_view_basename=None):
    """output_view_basename : ex. gold_ml_view_2022.csv pour generer une vue 2022 sans ecraser gold_ml_view.csv."""
    ensure_gold_dir()
    (elections, geography, etatcivil, etatcivil_dec, etatcivil_nais,
     deces_communes, nais_communes, emploi, chomage_dep, circo, population_dep,
     filosofi_com, diplomes, population_commune) = load_silver()

    pop, communes_10k, elec, geo_to_id, geo_df = _filter_communes_and_build_geo_tables(geography, elections)
    extra_parts = _load_extra_elections(communes_10k)
    voix_cols, nom_to_id_candidat = _build_candidats_and_partis(elec)
    opt_sources = {
        "etatcivil_dec": etatcivil_dec, "etatcivil_nais": etatcivil_nais,
        "deces_communes": deces_communes, "nais_communes": nais_communes,
        "emploi": emploi, "chomage_dep": chomage_dep,
        "filosofi_com": filosofi_com, "diplomes": diplomes, "population_commune": population_commune,
    }
    stat_per_geo, geo_to_id_stat = _build_stat_table(
        geography, communes_10k, geo_to_id, pop, elec, etatcivil, opt_sources
    )
    _write_resultat_and_vue_ml(elec, geo_df, stat_per_geo, voix_cols, nom_to_id_candidat,
                               geo_to_id_stat, geo_to_id, circo, population_dep, extra_elections_parts=extra_parts,
                               output_view_basename=output_view_basename)

    print(f"Gold genere : {len(geo_df)} communes (>={config.MIN_POPULATION} hab.)")
    print(f"  -> {config.GOLD_DIR}")
    if output_view_basename:
        print(f"  Vue ML : {output_view_basename}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=None, help="Annee election (defaut: config.ELECTION_YEAR)")
    p.add_argument("--min-pop", type=int, default=None, help="Population min commune (defaut: config.MIN_POPULATION)")
    p.add_argument("--output-view", type=str, default=None, metavar="FILE",
                   help="Nom du fichier vue ML (ex. gold_ml_view_2022.csv) pour ne pas ecraser gold_ml_view.csv")
    args = p.parse_args()
    if args.year is not None:
        config.ELECTION_YEAR = args.year
    if args.min_pop is not None:
        config.MIN_POPULATION = args.min_pop
    run(output_view_basename=args.output_view)
