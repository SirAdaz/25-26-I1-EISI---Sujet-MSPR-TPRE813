from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import config

DB_PATH = config.GOLD_DIR / "gold.db"

def get_conn():
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

# Replaces dataframe_to_dbf()
def save_table(df: pd.DataFrame, table_name: str):
    with get_conn() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

# Replaces dbf_to_dataframe()
def load_table(table_name: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

# Replaces load_gold_view() in train_predict_election.py
def load_gold_view() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Executer d'abord silver_to_gold.py. Fichier absent: {DB_PATH}")
    return load_table("gold_ml_view")