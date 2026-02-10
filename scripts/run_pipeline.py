# -*- coding: utf-8 -*-
"""
Point d'entree unique : execute le pipeline complet
Bronze -> Silver -> Gold -> Entrainement modele.
Parametres : --year, --min-pop, --target (candidat).
"""
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import config


def main():
    p = argparse.ArgumentParser(description="Pipeline medallion + IA predictions elections")
    p.add_argument("--year", type=int, default=config.ELECTION_YEAR, help="Annee election")
    p.add_argument("--min-pop", type=int, default=config.MIN_POPULATION, help="Population min par commune")
    p.add_argument("--target", default="MACRON", help="Candidat cible pour la cible ML (part des voix)")
    p.add_argument("--skip-train", action="store_true", help="Ne pas lancer l'entrainement du modele")
    args = p.parse_args()

    config.ELECTION_YEAR = args.year
    config.MIN_POPULATION = args.min_pop
    config.YEARS_FOR_ELECTION = [args.year - 1, args.year]

    print("Etape 1/3 : Bronze -> Silver")
    from bronze_to_silver import run as run_silver
    run_silver()

    print("\nEtape 2/3 : Silver -> Gold")
    from silver_to_gold import run as run_gold
    run_gold()

    if not args.skip_train:
        print("\nEtape 3/3 : Entrainement modele")
        from train_predict_election import run as run_train
        run_train(target_candidate=args.target)
    else:
        print("\nEtape 3/3 : Entrainement (ignore)")

    print("\nPipeline termine.")


if __name__ == "__main__":
    main()
