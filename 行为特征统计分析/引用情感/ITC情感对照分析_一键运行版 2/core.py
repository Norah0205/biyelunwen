import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config


def log(message):
    print(f"[ITC分析] {message}", flush=True)


def norm_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def norm_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def combo(a, b):
    return " || ".join(sorted((norm_text(a), norm_text(b))))


def _norm_number(value):
    """生成跨 CSV/Excel 稳定的数值文本（1、1.0 视为同值）。"""
    if pd.isna(value):
        return "<nan>"
    return format(float(value), ".15g")


def _record_digest(payload):
    return "R" + hashlib.blake2b(payload.encode("utf-8", errors="replace"), digest_size=16).hexdigest()


def enrich(df):
    """生成四象限、三元组聚类键和方向无关的记录级唯一键。"""
    x = df.copy()
    x["sbert相似度"] = pd.to_numeric(x["sbert相似度"], errors="coerce")
    x["耦合强度"] = pd.to_numeric(x["耦合强度"], errors="coerce")
    hs = x["sbert相似度"].ge(config.SBERT_THRESHOLD)
    hc = x["耦合强度"].ge(config.COUPLING_THRESHOLD)
    x["quadrant"] = np.select([hs & hc, hs & ~hc, ~hs & hc], ["Q11", "Q10", "Q01"], default="Q00")

    a = x["文献A_WOSID"].map(norm_id)
    d1, d2 = x["文献D1_WOSID"].map(norm_id), x["文献D2_WOSID"].map(norm_id)
    # 将每一侧的 D、题目、引用语境和参考文献绑定后整体换向，避免 D1/D2 交换改变 record_key。
    side1 = pd.DataFrame({
        "d": d1, "title": x["文献D1题目"].map(norm_text), "section": x.Section1.map(norm_text),
        "classification": x.Classification1.map(norm_text), "sentence": x["引用句1"].map(norm_text),
        "references": x["文献D1参考文献"].map(norm_text),
    }, index=x.index)
    side2 = pd.DataFrame({
        "d": d2, "title": x["文献D2题目"].map(norm_text), "section": x.Section2.map(norm_text),
        "classification": x.Classification2.map(norm_text), "sentence": x["引用句2"].map(norm_text),
        "references": x["文献D2参考文献"].map(norm_text),
    }, index=x.index)
    sep = "\x1f"
    s1 = side1.astype(str).agg(sep.join, axis=1)
    s2 = side2.astype(str).agg(sep.join, axis=1)
    swap = s1.gt(s2)
    left, right = s1.where(~swap, s2), s2.where(~swap, s1)
    lo, hi = d1.where(~swap, d2), d2.where(~swap, d1)

    x["pair_key"] = a + "|" + lo + "|" + hi
    metric = x["sbert相似度"].map(_norm_number) + sep + x["耦合强度"].map(_norm_number)
    payload = a + sep + x["文献A题目"].map(norm_text) + sep + left + sep + right + sep + metric
    x["record_key"] = payload.map(_record_digest)
    x["a_key"] = a
    x["section_combo"] = [combo(v1, v2) for v1, v2 in zip(x.Section1, x.Section2)]
    x["classification_combo"] = [combo(v1, v2) for v1, v2 in zip(x.Classification1, x.Classification2)]
    x["context_key"] = x.section_combo + " ## " + x.classification_combo
    return x


def stable_hash(text, seed=config.SEED):
    raw = f"{seed}|{text}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def normalized_sentence(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


def json_default(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)


def save_excel(path, sheets):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.iloc[:1_048_575].to_excel(writer, sheet_name=name[:31], index=False)
