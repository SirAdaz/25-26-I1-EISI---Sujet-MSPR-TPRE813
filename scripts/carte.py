"""
Génère une carte de France qui représente le parti politique en tête
à l'issue du 1er tour des présidentielles de 2017
Utilise le DataWarehouse du dossier gold
Si aucun jeu de donnée n'est présent, lancer la pipeline du modèle IA.
"""
import pandas as pd
import geopandas as gpd
import folium
import requests
from pathlib import Path

CSV_PATH     = "../gold/gold_ml_view.csv"
OUTPUT_HTML  = "../carte_elections_2017.html"
GEOJSON_PATH = "../communes.geojson"
GEOJSON_URL  = (
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson"
    "/master/communes-version-simplifiee.geojson"
)

CANDIDATS = {
    "part_voix_2017_LE_PEN"        : "Le Pen",
    "part_voix_2017_MACRON"        : "Macron",
    "part_voix_2017_FILLON"        : "Fillon",
    "part_voix_2017_M_LENCHON"     : "Mélenchon",
    "part_voix_2017_DUPONT_AIGNAN" : "Dupont-Aignan",
    "part_voix_2017_HAMON"         : "Hamon",
    "part_voix_2017_ASSELINEAU"    : "Asselineau",
    "part_voix_2017_ARTHAUD"       : "Arthaud",
    "part_voix_2017_POUTOU"        : "Poutou",
    "part_voix_2017_CHEMINADE"     : "Cheminade",
    "part_voix_2017_LASSALLE"      : "Lassalle",
}

COULEURS = {
    "Le Pen"        : "#004FA3",
    "Macron"        : "#FF8000",
    "Fillon"        : "#0070C0",
    "Mélenchon"     : "#CC0000",
    "Dupont-Aignan" : "#1A1A6E",
    "Hamon"         : "#FF69B4",
    "Asselineau"    : "#8B0000",
    "Arthaud"       : "#B22222",
    "Poutou"        : "#DC143C",
    "Cheminade"     : "#808080",
    "Lassalle"      : "#228B22",
}

def run():
    # ── Chargement SANS dtype=str sauf pour code_geo ──────────────
    df = pd.read_csv(CSV_PATH, sep=";", dtype={"code_geo": str})

    cols_candidats = list(CANDIDATS.keys())

    # Conversion numérique explicite
    for col in cols_candidats:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Vérification rapide
    print("Types après conversion :")
    print(df[cols_candidats].dtypes)
    print("\n3 premières lignes :")
    for _, row in df.head(3).iterrows():
        gagnant_col = df.loc[row.name, cols_candidats].idxmax()
        gagnant = CANDIDATS[gagnant_col]
        score = df.loc[row.name, cols_candidats].max() * 100
        print(f"  {row['ville']:30s} → {gagnant:15s} ({score:.1f}%)")

    sous_df = df[cols_candidats].copy()
    df["candidat"] = sous_df.idxmax(axis=1).map(CANDIDATS)
    df["score_max"] = (sous_df.max(axis=1) * 100).round(2)
    df["couleur"] = df["candidat"].map(COULEURS)

    print("\nRépartition :")
    print(df["candidat"].value_counts().to_string())

    # ── GeoJSON ───────────────────────────────────────────────────
    if not Path(GEOJSON_PATH).exists():
        r = requests.get(GEOJSON_URL, timeout=60)
        r.raise_for_status()
        Path(GEOJSON_PATH).write_bytes(r.content)

    gdf = gpd.read_file(GEOJSON_PATH)
    gdf = gdf.rename(columns={"code": "code_geo"})
    gdf["code_geo"] = gdf["code_geo"].str.strip().str.zfill(5)
    df["code_geo"] = df["code_geo"].astype(str).str.strip().str.zfill(5)

    cols_merge = [c for c in [
        "code_geo", "ville", "departement", "candidat", "score_max", "couleur",
        "pop_commune", "tauxChomage", "taux_pauvrette", "med_niveau_vie"
    ] if c in df.columns]

    gdf = gdf.merge(df[cols_merge], on="code_geo", how="left")

    # ── Carte ─────────────────────────────────────────────────────
    carte = folium.Map(location=[46.5, 2.5], zoom_start=6,
                       tiles="CartoDB positron", prefer_canvas=True)

    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda f: {
            "fillColor": f["properties"].get("couleur") or "#DDDDDD",
            "color": "#666", "weight": 0.3, "fillOpacity": 0.80,
        },
        highlight_function=lambda f: {"weight": 2, "color": "#000", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(
            fields=["ville", "departement", "candidat", "score_max"],
            aliases=["Commune", "Département", "Candidat en tête", "Score (%)"],
            localize=True, sticky=True,
        ),
    ).add_to(carte)

    compte = df["candidat"].value_counts()
    items = "".join(
        f'<div style="display:flex;align-items:center;margin:4px 0;">'
        f'<div style="width:14px;height:14px;background:{COULEURS[c]};border:1px solid #555;'
        f'border-radius:3px;margin-right:8px;"></div>'
        f'<span style="font-size:12px;">{c} <span style="color:#888;">({n})</span></span></div>'
        for c, n in compte.items() if c in COULEURS
    )
    carte.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:14px 18px;border-radius:8px;border:1px solid #ccc;
                box-shadow:2px 2px 8px rgba(0,0,0,.15);font-family:Arial,sans-serif;max-width:270px;">
      <b>️ Présidentielle 2017 — 1er tour</b>
      <div style="font-size:11px;color:#888;margin-bottom:8px;">
        Candidat en tête · communes &gt; 10 000 hab.<br>Gris = absent du jeu de données
      </div>{items}
    </div>"""))

    carte.save(OUTPUT_HTML)
    print(f"\n Carte générée : {OUTPUT_HTML}")

if __name__ == "__main__":
    run()