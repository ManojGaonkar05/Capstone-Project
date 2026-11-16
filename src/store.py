from pathlib import Path
import json
import pandas as pd


def load_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_jsonl(record, path):
    """Append one real generated-graph record to a JSON Lines dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
