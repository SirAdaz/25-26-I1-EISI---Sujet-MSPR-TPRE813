# Pipeline Medallion + IA predictions elections

Architecture Bronze -> Silver -> Gold alignee sur le MCD, puis modele IA pour predire les resultats d'election (part des voix) a partir des indicateurs par commune.

## Structure

- **bronze/** : donnees brutes (XLS elections, CSV geographie/delinquance, etat civil, Filosofi, diplomes, population)
- **silver/** : donnees nettoyees et filtrees par annee (genere par le script)
- **gold/** : tables MCD (geo, vote, DateDuVote, stat, candidat, parti_politique, situe, possede, resultat) + vue `gold_ml_view.csv`
- **models/** : modeles entraines (`.joblib`), metriques (`.meta.json`), graphiques et synthese (`summary_all_candidates.json` / `.png`)
- **scripts/** : `bronze_to_silver.py`, `silver_to_gold.py`, `train_predict_election.py`, `run_pipeline.py`
- **config.py** : `ELECTION_YEAR`, `MIN_POPULATION`, chemins

La vue Gold ML inclut, lorsqu’ils sont disponibles : indicateurs etat civil (taux mariage, deces, natalite), emploi/chomage, **Filosofi** (med_niveau_vie, taux_pauvrette), **diplomes** (part_sans_diplome, part_bac_plus), **population communale**, circo/region, et parts de voix des elections passees (2012, 2022). A chaque run d’entrainement, le script affiche quelles variables optionnelles sont presentes ou absentes.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Pipeline complet (Bronze -> Silver -> Gold -> entrainement)

```bash
python scripts/run_pipeline.py
```

Avec parametres (annee, population min, candidat cible) :

```bash
python scripts/run_pipeline.py --year 2017 --min-pop 10000 --target MACRON
python scripts/run_pipeline.py --skip-train   # sans entrainement du modele
```

### Etapes separees

```bash
python scripts/bronze_to_silver.py [--year 2017]
python scripts/silver_to_gold.py [--year 2017] [--min-pop 10000] [--output-view FICHIER]
python scripts/train_predict_election.py [--target MACRON] [options]
```

**silver_to_gold**

- `--year` : annee d’election (defaut : config).
- `--min-pop` : population minimale des communes (defaut : 10000).
- `--output-view FICHIER` : ecrit la vue ML dans ce fichier au lieu de `gold_ml_view.csv` (ex. `gold_ml_view_2022.csv` pour ne pas ecraser la vue 2017).

**train_predict_election**

- `--target CANDIDAT` : candidat cible (part des voix), ex. `MACRON` (defaut).
- `--all` : entrainer un modele pour chaque candidat puis generer la synthese JSON/PNG.
- `--model {rf,gb,xgb}` : type de modele (defaut : `gb` = Gradient Boosting).
- `--test-size` : part en test (defaut : 0.2).
- `--cv` / `--no-cv` : afficher ou non la cross-validation 5-fold (defaut : afficher).
- `--tune` : recherche d’hyperparametres (GridSearchCV, plus lent).
- `--no-train` : ne pas entrainer (chargement uniquement).
- `--top-features N` : n’utiliser que les N variables les plus importantes (reduit le bruit).
- `--holdout CHEMIN` : apres entrainement, evaluer le modele sur une vue Gold d’une autre annee (validation temporelle).

Exemples :

```bash
# Un candidat, 25 variables les plus importantes
python scripts/train_predict_election.py --target MACRON --top-features 25

# Tous les candidats + synthese
python scripts/train_predict_election.py --all

# Validation temporelle : entrainement 2017, evaluation sur vue 2022
python scripts/train_predict_election.py --target MACRON --holdout gold/gold_ml_view_2022.csv
```

### Validation temporelle (train 2017, test 2022)

Pour mesurer la stabilite du modele dans le temps :

1. Generer la vue Gold 2022 sans ecraser la vue 2017 (apres avoir prepare le Silver 2022, ex. `elections_2022.csv`, etc.) :

   ```bash
   python scripts/silver_to_gold.py --year 2022 --output-view gold_ml_view_2022.csv
   ```

2. Entrainer sur 2017 et evaluer sur 2022 :

   ```bash
   python scripts/train_predict_election.py --target MACRON --holdout gold/gold_ml_view_2022.csv
   ```

La vue holdout doit avoir la meme structure que `gold_ml_view.csv` (memes colonnes features + `voix_<CANDIDAT>` pour la cible).

## Fichiers IVX/IVD (Beyond 20/20)

Les donnees INSEE au format Beyond (.ivx, .ivd) ne sont pas lisibles nativement. Pour les utiliser (deces, naissances, enquete emploi), exporter en CSV depuis Beyond 20/20 Browser puis placer les CSV dans `bronze/` ; les scripts CSV les prendront en charge.
