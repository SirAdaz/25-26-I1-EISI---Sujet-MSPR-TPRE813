# -*- coding: utf-8 -*-
"""
Entrainement et prediction : modele IA pour predire les resultats d'election
a partir des indicateurs Gold (tauxSec, TauxMariage, etc.).
Cible par defaut : part des voix du candidat en tete (ex. MACRON 2017).
Parametrable pour d'autres elections et cibles.

Structure du script (pour s'y retrouver) :
  1. Donnees : chargement Gold, variables derivees (2012), preparation X/y
  2. Modeles : factory (rf/gb/xgb), train, CV, tuning
  3. Evaluation : accuracy +/- N pt, importance variables, sauvegarde, graphique
  4. Run : verification Gold, top-features, entrainement, holdout temporel
  5. Synthese : --all -> summary JSON + PNG
  6. CLI : main() et arguments
"""
from pathlib import Path
import sys
import argparse
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


# -----------------------------------------------------------------------------
# Donnees : chargement Gold, variables derivees, preparation X / y
# -----------------------------------------------------------------------------

def load_gold_view():
    """Charge la vue Gold pour le ML."""
    path = config.GOLD_DIR / "gold_ml_view.csv"
    if not path.exists():
        raise FileNotFoundError(f"Executer d'abord silver_to_gold.py. Fichier absent: {path}")
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    return df


def add_derived_features(df):
    """
    Ajoute des variables derivees (ratios, agregats 2012) pour ameliorer le pouvoir predictif.
    Modifie df en place, retourne la liste des noms de colonnes ajoutees.
    """
    added = []
    eps = 1e-6  # eviter division par zero dans les ratios
    # Noms des colonnes part_voix presidentielle 2012 (pour ratios gauche/droite/centre)
    sarko = "part_voix_SARKOZY_2012_presidentielle"
    lepen = "part_voix_LE_PEN_2012_presidentielle"
    hollande = "part_voix_HOLLANDE_2012_presidentielle"
    melenchon = "part_voix_M_LENCHON_2012_presidentielle"
    bayrou = "part_voix_BAYROU_2012_presidentielle"
    for c in [sarko, lepen, hollande, melenchon, bayrou]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Ratio droite 2012 (Sarko/Le Pen) et ratio gauche/droite
    if sarko in df.columns and lepen in df.columns:
        df["_ratio_sarko_lepen_2012"] = df[sarko] / (df[lepen].replace(0, np.nan).fillna(eps) + eps)
        added.append("_ratio_sarko_lepen_2012")
    if hollande in df.columns and melenchon in df.columns and sarko in df.columns and lepen in df.columns:
        gauche = df[hollande].fillna(0) + df[melenchon].fillna(0)
        droite = df[sarko].fillna(0) + df[lepen].fillna(0)
        df["_ratio_gauche_droite_2012"] = gauche / (droite + eps)
        added.append("_ratio_gauche_droite_2012")
    # Parts agrégées 2012 (droite = Sarko+LP, gauche = Hollande+Mélenchon, centre = Bayrou)
    if sarko in df.columns and lepen in df.columns:
        df["_part_droite_2012"] = df[sarko].fillna(0) + df[lepen].fillna(0)
        added.append("_part_droite_2012")
    if hollande in df.columns and melenchon in df.columns:
        df["_part_gauche_2012"] = df[hollande].fillna(0) + df[melenchon].fillna(0)
        added.append("_part_gauche_2012")
    if bayrou in df.columns:
        df["_part_centre_2012"] = df[bayrou].fillna(0)
        added.append("_part_centre_2012")
    # Nettoyer inf (division par zero)
    for c in added:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return added


def prepare_xy(df, target_candidate="MACRON", feature_cols=None):
    """
    Prepare les matrices X (features) et y (cible).
    Cible = part des voix du candidat (pourcentage). Utilise part_voix_2017_<CANDIDAT> si presente,
    sinon voix_<CANDIDAT> / total exprimés (vue ancienne).
    """
    df = df.copy()
    target_part_col = f"part_voix_2017_{target_candidate}"
    if target_part_col in df.columns:
        df["_target"] = pd.to_numeric(df[target_part_col], errors="coerce")
    else:
        voix_cols = [c for c in df.columns if c.startswith("voix_")]
        if not voix_cols:
            raise ValueError("Aucune colonne part_voix_2017_* ni voix_* dans la vue Gold.")
        target_col = f"voix_{target_candidate}"
        if target_col not in df.columns:
            raise ValueError(f"Colonne {target_col} absente. Candidats: {[c.replace('voix_','') for c in voix_cols]}")
        df["_exprimes"] = df[voix_cols].sum(axis=1).replace(0, np.nan)
        df["_target"] = df[target_col] / df["_exprimes"]
    df = df.dropna(subset=["_target"])

    # Si pas de liste fournie : indicateurs de base + part_voix (elections passées, pas année courante = fuite)
    if feature_cols is None:
        feature_cols = [c for c in [
            "tauxChomage", "tauxSec", "TauxMariage", "TauxDec", "tauxNatalite",
            "pop_dep", "circo", "REG"
        ] if c in df.columns]
        part_voix_cols = [
            c for c in df.columns
            if c.startswith("part_voix_") and not c.startswith(f"part_voix_{config.ELECTION_YEAR}_")
        ]
        feature_cols = feature_cols + part_voix_cols
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # part_voix manquantes = 0 (commune absente d'une election passee)
    part_cols = [c for c in feature_cols if c.startswith("part_voix_")]
    if part_cols:
        df[part_cols] = df[part_cols].fillna(0)
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    y = df["_target"]
    return X, y, df


# -----------------------------------------------------------------------------
# Modeles : factory (rf / gb / xgb), entrainement, cross-validation, tuning
# -----------------------------------------------------------------------------

def _make_model(model_name="rf", random_state=42, **kwargs):
    """Factory : Random Forest, Gradient Boosting ou XGBoost (si dispo). Fallback RF si xgb/gb absent."""
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        except ImportError:
            print("  (xgboost non installe: pip install xgboost)")
            model_name = "gb"
    if model_name == "gb" or model_name == "xgboost":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=180,
                max_depth=6,
                min_samples_leaf=4,
                learning_rate=0.08,
                random_state=random_state,
                **kwargs
            )
        except ImportError:
            pass
    return RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        min_samples_leaf=5,
        max_features="sqrt",
        **kwargs
    )


def accuracy_within_tolerance(y_true, y_pred, tolerances=(0.01, 0.02, 0.03)):
    """
    Pouvoir predictif : part des predictions a moins de X points de la part reelle.
    Retourne un dict ex. {"within_1pt": 0.45, "within_2pt": 0.78, "within_3pt": 0.92}.
    """
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    out = {}
    for tol in tolerances:
        key = f"accuracy_within_{int(tol*100)}pt"
        out[key] = float(np.mean(err <= tol))
    return out


def train_model(X, y, test_size=0.2, random_state=42, model_name="rf", **kwargs):
    """Entraine le modele et retourne (model, metrics, (X_test, y_test, y_pred))."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = _make_model(model_name=model_name, random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }
    metrics.update(accuracy_within_tolerance(y_test, y_pred))
    return model, metrics, (X_test, y_test, y_pred)


def cross_validate_model(X, y, cv=5, model_name="rf", random_state=42):
    """Cross-validation pour une estimation plus stable du R2 et MAE."""
    from sklearn.model_selection import cross_validate
    model = _make_model(model_name=model_name, random_state=random_state)
    scoring = ["r2", "neg_mean_absolute_error"]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    r2_mean = res["test_r2"].mean()
    mae_mean = -res["test_neg_mean_absolute_error"].mean()
    return {"R2_cv_mean": r2_mean, "R2_cv_std": res["test_r2"].std(), "MAE_cv_mean": mae_mean}


def tune_and_train(X, y, test_size=0.2, random_state=42, model_name="xgb"):
    """GridSearchCV sur Gradient Boosting, refit du meilleur modele, evaluation sur le split test."""
    from sklearn.model_selection import GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    base = _make_model(model_name=model_name, random_state=random_state)
    param_grid = {
        "n_estimators": [120, 150, 180],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.06, 0.08, 0.1],
        "min_samples_leaf": [3, 4, 5],
    }
    grid = GridSearchCV(
        base, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }
    metrics.update(accuracy_within_tolerance(y_test, y_pred))
    print(f"  Tuning: meilleurs params = {grid.best_params_}")
    return model, metrics, (X_test, y_test, y_pred)


# -----------------------------------------------------------------------------
# Evaluation : metriques, importance variables, sauvegarde modele, graphique
# -----------------------------------------------------------------------------

def print_feature_importance(model, feature_cols):
    """Affiche les 15 variables les plus importantes (RandomForest ou GradientBoosting)."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("  Top 15 variables :")
    for name, val in imp.head(15).items():
        print(f"    {val:.3f}  {name}")


def save_model(model, feature_cols, target_candidate, metrics, path=None):
    path = path or config.MODELS_DIR / f"model_{config.ELECTION_YEAR}_{target_candidate}.joblib"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"model": model, "feature_cols": feature_cols, "target": target_candidate}, path)
    except ImportError:
        path = path.with_suffix(".json")
        # Sauvegarde simplifiee sans le modele (sklearn n'est pas serialisable en JSON)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"feature_cols": feature_cols, "target": target_candidate, "metrics": metrics}, f, indent=2)
        print(f"(joblib non installe : metriques sauvees dans {path})")
        return
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"election_year": config.ELECTION_YEAR, "target": target_candidate, "metrics": metrics}, f, indent=2)
    print(f"Modele sauve : {path}")


def plot_predictions(y_test, y_pred, target_candidate, metrics, save_path=None):
    """
    Trace un graphique Reel vs Predicted (nuage de points + droite parfaite).
    Sauvegarde dans models/prediction_plot_ANNEE_CANDIDAT.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib non installe : pas de graphique)")
        return
    save_path = save_path or config.MODELS_DIR / f"prediction_plot_{config.ELECTION_YEAR}_{target_candidate}.png"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.6, s=25, label="Communes (test)")
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2, label="Prediction parfaite")
    ax.set_xlabel("Part des voix reelle")
    ax.set_ylabel("Part des voix predite")
    ax.set_title(f"Qualite des predictions - {target_candidate} ({config.ELECTION_YEAR})\n"
                 f"MAE={metrics['MAE']:.3f}  R2={metrics['R2']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Graphique sauve : {save_path}")


# -----------------------------------------------------------------------------
# Run : verification Gold, construction features, top-features, train, holdout
# -----------------------------------------------------------------------------

# Variables optionnelles (Filosofi, diplômes, pop communale) : vérifiées à chaque run
OPTIONAL_GOLD_VARS = ["med_niveau_vie", "taux_pauvrette", "part_sans_diplome", "part_bac_plus", "pop_commune"]


def _check_optional_gold_vars(df):
    """Affiche quelles variables optionnelles (Filosofi, diplômes, pop) sont présentes dans la vue Gold."""
    present = [v for v in OPTIONAL_GOLD_VARS if v in df.columns]
    absent = [v for v in OPTIONAL_GOLD_VARS if v not in df.columns]
    print("Vue Gold – variables optionnelles (Filosofi, diplômes, pop communale):")
    print(f"  Présentes: {present if present else 'aucune'}")
    if absent:
        print(f"  Absentes:  {absent} (relancer silver_to_gold si besoin)")


def run(train=True, target_candidate="MACRON", test_size=0.2, model_name="xgb", do_cv=False, do_tune=False,
       top_features=None, holdout_path=None):
    df = load_gold_view()
    _check_optional_gold_vars(df)
    derived = add_derived_features(df)
    # Liste des features : indicateurs de base + optionnels (Filosofi, diplômes, pop) + dérivées + part_voix
    feature_cols = [c for c in [
        "tauxChomage", "tauxSec", "TauxMariage", "TauxDec", "tauxNatalite",
        "pop_dep", "circo", "REG",
        "med_niveau_vie", "taux_pauvrette", "part_sans_diplome", "part_bac_plus", "pop_commune",
    ] if c in df.columns]
    feature_cols = feature_cols + derived
    # Part des voix des elections PASSÉES uniquement (pas 2017 : ce serait de la fuite de donnees / triche)
    part_voix_cols = [
        c for c in df.columns
        if c.startswith("part_voix_") and not c.startswith(f"part_voix_{config.ELECTION_YEAR}_")
    ]
    feature_cols = feature_cols + part_voix_cols
    X, y, _ = prepare_xy(df, target_candidate=target_candidate, feature_cols=feature_cols)
    print(f"Echantillons: {len(X)}, features: {len(feature_cols)} (dont {len(part_voix_cols)} part_voix passées, {len(derived)} dérivées), cible: part_voix_2017_{target_candidate}")

    # Optionnel : garder seulement les N variables les plus importantes (1er fit -> importances -> refit)
    if top_features and top_features > 0 and len(feature_cols) > top_features:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42)
        temp_model = _make_model(model_name=model_name, random_state=42)
        temp_model.fit(X_train, y_train)
        imp = pd.Series(temp_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        feature_cols = imp.head(top_features).index.tolist()
        X = X[feature_cols]
        print(f"Top-features: utilisation des {len(feature_cols)} variables les plus importantes.")

    if do_cv:
        cv_metrics = cross_validate_model(X, y, cv=5, model_name=model_name)
        print(f"  Cross-validation (5-fold): R2 = {cv_metrics['R2_cv_mean']:.4f} (+/- {cv_metrics['R2_cv_std']:.4f}), MAE = {cv_metrics['MAE_cv_mean']:.4f}")

    if train:
        return _do_train_and_save(
            X, y, feature_cols, target_candidate, test_size, model_name,
            do_tune, holdout_path
        )
    return None, feature_cols


def _do_train_and_save(X, y, feature_cols, target_candidate, test_size, model_name, do_tune, holdout_path):
    """Entrainement (avec ou sans tuning), affichage metriques, sauvegarde joblib+meta, graphique, puis holdout si demande."""
    if do_tune and model_name == "gb":
        model, metrics, (_, y_test, y_pred) = tune_and_train(
            X, y, test_size=test_size, model_name=model_name
        )
    else:
        model, metrics, (_, y_test, y_pred) = train_model(
            X, y, test_size=test_size, model_name=model_name
        )
    print(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")
    y_std = float(y.std())
    if y_std < 0.02:
        print("  (Attention: part des voix tres faible et peu variable -> accuracy +/- 1 a 3 pt peu interpretable, privilegier MAE/R2)")
    print("  Pouvoir predictif (Accuracy):")
    # Proportion de communes ou l'erreur absolue est <= 1, 2 ou 3 points
    for pts, k in [(1, "accuracy_within_1pt"), (2, "accuracy_within_2pt"), (3, "accuracy_within_3pt")]:
        if k in metrics:
            pct_ok = metrics[k] * 100
            pct_ko = 100 - pct_ok
            print(f"    +/- {pts} pt : a raison {pct_ok:.1f}% | se trompe {pct_ko:.1f}%")
    print_feature_importance(model, feature_cols)
    save_model(model, feature_cols, target_candidate, metrics)
    plot_predictions(y_test, y_pred, target_candidate, metrics)
    if holdout_path:
        evaluate_holdout(model, feature_cols, holdout_path, target_candidate)
    return model, feature_cols


def evaluate_holdout(model, feature_cols, holdout_path, target_candidate):
    """
    Validation temporelle : évalue le modèle sur une vue Gold d'une autre année (ex. 2022).
    holdout_path : chemin vers un CSV même format que gold_ml_view (features + part_voix_2017_* pour la cible).
    """
    path = Path(holdout_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"  Holdout ignoré : fichier introuvable {path}")
        return
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    add_derived_features(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Holdout ignoré : colonnes manquantes dans la vue {path.name}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return
    target_col = f"part_voix_2017_{target_candidate}"
    if target_col not in df.columns and f"voix_{target_candidate}" not in df.columns:
        print(f"  Holdout ignoré : colonne cible {target_col} (ou voix_{target_candidate}) absente dans {path.name}")
        return
    x_holdout, y_holdout, _ = prepare_xy(df, target_candidate=target_candidate, feature_cols=feature_cols)
    if len(x_holdout) == 0:
        print("  Holdout : aucun échantillon valide après dropna.")
        return
    y_pred = model.predict(x_holdout)
    r2 = float(r2_score(y_holdout, y_pred))
    mae = float(mean_absolute_error(y_holdout, y_pred))
    print(f"  Validation temporelle (holdout {path.name}) : R2 = {r2:.4f}, MAE = {mae:.4f} (n = {len(x_holdout)} communes)")


# -----------------------------------------------------------------------------
# Synthese : liste candidats, agregation metriques (--all), JSON + PNG
# -----------------------------------------------------------------------------

def get_all_candidates():
    """Retourne la liste des noms de candidats (part_voix_2017_* ou voix_*) dans la vue Gold."""
    df = load_gold_view()
    part_cols = [c for c in df.columns if c.startswith(f"part_voix_{config.ELECTION_YEAR}_")]
    if part_cols:
        return sorted([c.replace(f"part_voix_{config.ELECTION_YEAR}_", "") for c in part_cols])
    voix_cols = [c for c in df.columns if c.startswith("voix_")]
    return sorted([c.replace("voix_", "") for c in voix_cols])


def _collect_metrics_from_meta_files():
    """Lit tous les .meta.json des modeles sauves et retourne { candidat: { metrics } }."""
    import glob
    summary = {"election_year": config.ELECTION_YEAR, "candidates": {}}
    pattern = str(config.MODELS_DIR / "model_*_*.joblib.meta.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cand = data.get("target")
            if cand and "metrics" in data:
                summary["candidates"][cand] = data["metrics"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return summary


def write_summary_all_candidates():
    """Agrege les metriques de tous les candidats dans un JSON + un PNG recapitulatif."""
    summary = _collect_metrics_from_meta_files()
    if not summary["candidates"]:
        print("  (Aucun fichier .meta.json trouve, pas de synthese generee)")
        return
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.MODELS_DIR / "summary_all_candidates.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Synthese JSON : {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    candidates = list(summary["candidates"].keys())
    r2 = [summary["candidates"][c].get("R2", 0) for c in candidates]
    mae = [summary["candidates"][c].get("MAE", 0) for c in candidates]
    acc2 = [summary["candidates"][c].get("accuracy_within_2pt", 0) * 100 for c in candidates]
    # 3 graphiques en barres horizontales : R2, MAE, % within +/- 2 pt
    _, axes = plt.subplots(1, 3, figsize=(12, max(5, len(candidates) * 0.35)))
    axes[0].barh(candidates, r2, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("R2")
    axes[0].set_title("R2 par candidat")
    axes[0].set_xlim(0, 1)
    axes[1].barh(candidates, mae, color="coral", alpha=0.8)
    axes[1].set_xlabel("MAE")
    axes[1].set_title("MAE par candidat")
    axes[2].barh(candidates, acc2, color="seagreen", alpha=0.8)
    axes[2].set_xlabel("%")
    axes[2].set_title("Accuracy within +/- 2 pt")
    axes[2].set_xlim(0, 105)
    plt.suptitle(f"Metriques par candidat - Election {config.ELECTION_YEAR}", fontsize=12)
    plt.tight_layout()
    png_path = config.MODELS_DIR / "summary_all_candidates.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Synthese PNG  : {png_path}")


# -----------------------------------------------------------------------------
# CLI : arguments et point d'entree
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entrainement / prediction elections")
    parser.add_argument("--target", default="MACRON", help="Candidat cible (part des voix)")
    parser.add_argument("--all", action="store_true", help="Entrainer un modele pour chaque candidat (un par un)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Part des donnees en test")
    parser.add_argument("--model", default="xgb", choices=["rf", "gb", "xgb"], help="xgb=XGBoost (defaut), gb=GradientBoosting, rf=RandomForest")
    parser.add_argument("--cv", action="store_true", default=True, help="Afficher R2/MAE en cross-validation 5-fold (estimation stable, defaut: True)")
    parser.add_argument("--no-cv", action="store_true", help="Ne pas afficher la cross-validation")
    parser.add_argument("--tune", action="store_true", help="Recherche des meilleurs hyperparametres (plus performant mais lent)")
    parser.add_argument("--no-train", action="store_true", help="Ne pas entrainer (charger seulement)")
    parser.add_argument("--top-features", type=int, default=None, metavar="N",
                        help="N'utiliser que les N variables les plus importantes (reduit le bruit)")
    parser.add_argument("--holdout", type=str, default=None, metavar="PATH",
                        help="Chemin vers une vue Gold d'une autre annee (ex. gold/gold_ml_view_2022.csv) pour validation temporelle")
    args = parser.parse_args()

    do_cv = args.cv and not getattr(args, "no_cv", False)
    # Mode --all : un modèle par candidat puis synthèse JSON + PNG
    if args.all:
        candidates = get_all_candidates()
        print(f"Entrainement pour {len(candidates)} candidats : {candidates}\n")
        for i, cand in enumerate(candidates, 1):
            print(f"========== [{i}/{len(candidates)}] Candidat: {cand} ==========")
            run(train=not args.no_train, target_candidate=cand, test_size=args.test_size,
                model_name=args.model, do_cv=do_cv, do_tune=args.tune,
                top_features=args.top_features, holdout_path=args.holdout)
            print()
        if not args.no_train:
            print("========== Synthese tous candidats ==========")
            write_summary_all_candidates()
        return

    # Un seul candidat cible
    run(train=not args.no_train, target_candidate=args.target, test_size=args.test_size,
        model_name=args.model, do_cv=do_cv, do_tune=args.tune,
        top_features=args.top_features, holdout_path=args.holdout)


if __name__ == "__main__":
    main()
