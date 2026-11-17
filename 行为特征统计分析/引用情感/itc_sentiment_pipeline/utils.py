import hashlib
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABELS, PAIR_REQUIRED_COLUMNS, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(value):
    """Normalize citation sentences while retaining semantic words."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\[[0-9,;\-–—\s]+\]", " ", text)
    text = re.sub(r"\([0-9,;\-–—\s]+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_id(text):
    return "S_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def canonical_label(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    aliases = {
        "neg": "negative", "negative": "negative", "-1": "negative",
        "neu": "neutral", "neutral": "neutral", "0": "neutral",
        "pos": "positive", "positive": "positive", "1": "positive",
        "uncertain": "uncertain", "不确定": "uncertain",
    }
    return aliases.get(value, value)


def validate_pair_data(df):
    missing = [c for c in PAIR_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Input contains no records")
    for col in ["引用句1", "引用句2"]:
        if df[col].isna().any():
            raise ValueError(f"{col} contains {int(df[col].isna().sum())} missing values")
    if (pd.to_numeric(df["耦合强度"], errors="coerce") < 10).any():
        raise ValueError("Found records with bibliographic coupling strength < 10")
    if (pd.to_numeric(df["sbert相似度"], errors="coerce") < 0.60).any():
        raise ValueError("Found records with SBERT similarity < 0.60")


def prepare_sentence_inventory(pair_df):
    frames = []
    for side in [1, 2]:
        tmp = pd.DataFrame({
            "pair_row_id": np.arange(1, len(pair_df) + 1),
            "side": side,
            "source_col": f"引用句{side}",
            "text": pair_df[f"引用句{side}"].astype(str),
        })
        frames.append(tmp)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["norm_key"] = long_df["text"].map(normalize_text)
    if (long_df["norm_key"].str.len() == 0).any():
        raise ValueError("Normalization created empty citation sentences")
    long_df["sent_id"] = long_df["norm_key"].map(stable_id)

    inventory = (
        long_df.groupby(["sent_id", "norm_key"], as_index=False)
        .agg(text=("text", "first"), occurrence=("pair_row_id", "size"))
    )
    inventory["char_len"] = inventory["text"].str.len()
    return long_df, inventory


def load_training_corpus(path: Path, text_col: str, label_col: str):
    if not path.exists():
        raise FileNotFoundError(f"Training corpus not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Training corpus must be CSV or Excel")
    if text_col not in df or label_col not in df:
        raise ValueError(f"Training corpus requires columns: {text_col}, {label_col}")
    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()
    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].map(canonical_label)
    out = out[out["label"].isin(LABELS)]
    out = out[out["text"].str.len().between(20, 2000)]
    out["norm_key"] = out["text"].map(normalize_text)
    out = out.drop_duplicates("norm_key").reset_index(drop=True)
    return out


def save_excel_sheets(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, data in sheets.items():
            data.to_excel(writer, sheet_name=name[:31], index=False)
