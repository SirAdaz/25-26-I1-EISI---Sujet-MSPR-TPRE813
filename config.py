# -*- coding: utf-8 -*-
"""
Configuration du pipeline medallion et de l'IA de prédiction.
Paramétrable pour d'autres élections (année, seuil population, etc.).

Structure Bronze attendue (voir scripts/reorganize_bronze.py) :
  bronze/elections/   geographie/   etat_civil/   emploi_chomage/   population/   circonscriptions/
"""
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent

# Chemins des couches
BRONZE_DIR = PROJECT_ROOT / "bronze"
SILVER_DIR = PROJECT_ROOT / "silver"
GOLD_DIR = PROJECT_ROOT / "gold"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

# Paramètres élection (ex. présidentielle 2017)
ELECTION_YEAR = 2017
ELECTION_LABEL = "Presidentielle_2017_Tour_1"

# Filtre communes pour la couche Gold (population minimale)
MIN_POPULATION = 10_000

# Années de données à garder pour les indicateurs (autour de l'élection)
YEARS_FOR_ELECTION = [2016, 2017]

# --- Bronze : structure organisée (dossiers par thème) ---
# Pour créer la structure : python scripts/reorganize_bronze.py
def _bronze(path_new, path_old=None):
    """Utilise le chemin nouveau si présent, sinon l'ancien (compatibilité)."""
    p = BRONZE_DIR / path_new if isinstance(path_new, str) else path_new
    if p.exists():
        return p
    if path_old is not None:
        old = BRONZE_DIR / path_old if isinstance(path_old, str) else path_old
        if old.exists():
            return old
    return p

# Élections (résultats par commune)
BRONZE_ELECTIONS_XLS = _bronze("elections/presidentielle_2017_tour1.xls", "Presidentielle_2017_Resultats_Communes_Tour_1_c.xls")
BRONZE_ELECTIONS_EXTRA = [
    {"year": 2012, "type": "presidentielle", "path": _bronze("elections/presidentielle_2012.xls", "presidentielleResultats2012.xls")},
    {"year": 2012, "type": "legislative", "path": _bronze("elections/legislative_2012.xls", "legislativesResultat2012.xls")},
    {"year": 2022, "type": "legislative", "path": _bronze("elections/legislative_2022_tour1_communes.xlsx", "resultats-par-niveau-subcom-t1-france-entiere2022.xlsx")},
]
# Géographie et délinquance
BRONZE_GEOGRAPHY_CSV = _bronze("geographie/geographie_delinquance.csv", "donnee-data.gouv-2024-geographie2025-produit-le2025-06-04.csv/donnee-data.gouv-2024-geographie2025-produit-le2025-06-04.csv")
# État civil 2017
BRONZE_ETATCIVIL_MAR_CSV = _bronze("etat_civil/2017/mariages_2017.csv", "2017/etatcivil2017_mar2017_csv/etatcivil2017_mar2017.csv")
BRONZE_ETATCIVIL_DEC_DBF = _bronze("etat_civil/2017/deces_2017.dbf", "2017/etatcivil2017_dec2017_dbase/dec2017.dbf")
BRONZE_ETATCIVIL_NAIS_DBF = _bronze("etat_civil/2017/naissances_2017.dbf", "2017/etatcivil2017_nais2017_dbase/nais2017.dbf")
# État civil par commune (data.gouv)
BRONZE_DECES_COMMUNES_CSV = _bronze("etat_civil/deces_communes.csv", "DS_ETAT_CIVIL_DECES_COMMUNES_CSV_FR/DS_ETAT_CIVIL_DECES_COMMUNES_data.csv")
BRONZE_NAIS_COMMUNES_CSV = _bronze("etat_civil/naissances_communes.csv", "DS_ETAT_CIVIL_NAIS_COMMUNES_CSV_FR/DS_ETAT_CIVIL_NAIS_COMMUNES_data.csv")
# Emploi / chômage (enquête emploi par année : DBF ou CSV)
BRONZE_EMPLOI_DBF = _bronze("emploi_chomage/enquete_emploi_2017.dbf", "chomage2017/fdeec17.dbf")
if not BRONZE_EMPLOI_DBF.exists():
    BRONZE_EMPLOI_DBF = _bronze("emploi_chomage/enquete_emploi_2017.dbf", "fd_eec17_dbase/fdeec17.dbf")
# Fichiers emploi par année (pour Silver emploi_YYYY.csv)
BRONZE_EMPLOI_BY_YEAR = {
    2017: BRONZE_EMPLOI_DBF,
    2018: BRONZE_DIR / "chomage2018" / "FD_dbf_EEC18.dbf",
    2019: BRONZE_DIR / "chomage2019" / "FD_EEC_2019.dbf",
    2020: BRONZE_DIR / "chomage2020" / "FD_EEC_2020.dbf",
    2021: BRONZE_DIR / "chomage2021" / "FD_EEC_2021.dbf",
    2022: BRONZE_DIR / "chomage2022" / "FD_EEC_2022.dbf",
    2023: BRONZE_DIR / "chomage2023" / "FD_csv_EEC23.csv",
}
# Taux de chômage par département (Insee, trimestriel) — dans emploi_chomage après réorganisation
BRONZE_CHOMAGE_DEP_DIR = _bronze("emploi_chomage/taux_chomage_dep", "famille_TAUX-CHOMAGE_10022026")
def _chomage_dep_path(name_end: str):
    d = BRONZE_CHOMAGE_DEP_DIR
    if not d.exists():
        return None
    for f in d.iterdir():
        if f.name.endswith(name_end) or name_end in f.name:
            return f
    return d / name_end
BRONZE_CHOMAGE_DEP_CARACT = _chomage_dep_path("caract")  # caractéristiques.csv
BRONZE_CHOMAGE_DEP_VALEURS = _chomage_dep_path("valeurs_trimestrielles.csv")
# Population, circonscriptions
BRONZE_POPULATION_CSV = _bronze("population/estimation_population_departements.csv", "DS_ESTIMATION_POPULATION_CSV_FR/DS_ESTIMATION_POPULATION_data.csv")
# Population communale (historiques par commune et année)
BRONZE_POP_HISTORIQUES_CSV = _bronze(
    "population/populations_historiques/DS_POPULATIONS_HISTORIQUES_data.csv",
    "DS_POPULATIONS_HISTORIQUES_CSV_FR/DS_POPULATIONS_HISTORIQUES_data.csv",
)
# Revenus / niveau de vie (Filosofi par commune)
BRONZE_FILOSOFI_COM_CSV = _bronze(
    "revenus/filosofi_2017/cc_filosofi_2017_COM.CSV",
    "base-filosofi-2017_CSV/cc_filosofi_2017_COM.CSV",
)
# Diplômes (recensement, par commune)
BRONZE_DIPLOMES_CSV = _bronze(
    "diplomes/recensement_2022/DS_RP_DIPLOMES_PRINC_2022_data.csv",
    "DS_RP_DIPLOMES_PRINC_2022_CSV_FR/DS_RP_DIPLOMES_PRINC_2022_data.csv",
)
BRONZE_CIRCO_XLSX = _bronze("circonscriptions/circo_composition.xlsx", "circo_composition.xlsx")
