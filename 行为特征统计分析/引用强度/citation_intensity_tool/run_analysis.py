# -*- coding: utf-8 -*-
"""ITC+非ITC四组引用强度一键分析；兼容 Python 3.9。"""
import argparse
import html
import logging
import math
import re
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("/Users/bytedance/Desktop/1015引用强度")
DEFAULT_PAIR = DEFAULT_DIR / "配对关系-SBERT.csv"
DEFAULT_CITATION = DEFAULT_DIR / "引用内容.xlsx"
SEED = 20261015
SIM_CUT = 0.60
COUPLING_CUT = 10
FUZZY_CUT = 0.86

try:
    from tqdm.auto import tqdm
except ImportError:  # requirements未安装完整时仍可运行，并自动退化为普通日志
    tqdm = None


class ProgressReporter:
    """TTY中显示简洁进度条；重定向/PyCharm无TTY时输出带时间戳日志。"""
    def __init__(self, logger=None):
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)()) and tqdm is not None
        self.logger = logger or logging.getLogger(__name__)
        self.stage_name = "初始化"
        self.stage_started = time.monotonic()

    @staticmethod
    def _stamp():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def stage(self, percent, name):
        self.stage_name = name
        self.stage_started = time.monotonic()
        message = "[%3d%%] %s" % (percent, name)
        # 只通过 logger 输出，避免非TTY环境同时 logger+print 导致重复刷屏。
        self.logger.info(message)
        if self.tty:
            tqdm.write(message)

    def iter(self, iterable, desc, total=None):
        if self.tty:
            return tqdm(iterable, total=total, desc=desc, unit="项", dynamic_ncols=True,
                        mininterval=0.5, leave=False)
        return iterable

    @contextmanager
    def heartbeat(self, message, interval=15):
        """向量化步骤不伪造行级百分比；仅在耗时较长时定期报告已用时间。"""
        stop = threading.Event()
        def beat():
            while not stop.wait(interval):
                elapsed = time.monotonic() - self.stage_started
                text = "%s（已用时 %.0f 秒，仍在处理）" % (message, elapsed)
                self.logger.info(text)
        worker = threading.Thread(target=beat, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=0.2)

ALIASES = {
    "A_id": ["文献A_WOSID", "A_WOSID", "文献AID", "AID", "A编号"],
    "B_id": ["文献D1_WOSID", "D1_WOSID", "文献D1ID", "D1ID", "BID"],
    "C_id": ["文献D2_WOSID", "D2_WOSID", "文献D2ID", "D2ID", "CID"],
    "A_title": ["文献A_题目", "文献A题目", "A题目", "A标题", "文献A_title", "A_title"],
    "B_title": ["文献D1_题目", "文献D1题目", "D1题目", "D1标题", "文献D1_title", "D1_title"],
    "C_title": ["文献D2_题目", "文献D2题目", "D2题目", "D2标题", "文献D2_title", "D2_title"],
    "coupling": ["耦合强度", "文献耦合强度", "coupling", "coupling_strength", "共被引强度"],
    "similarity": ["sbert相似度", "SBERT相似度", "sbert_similarity", "similarity", "语义相似度"],
    "citing_id": ["文献D_WOSID", "施引文献WOSID", "施引文献ID", "citing_id", "source_id"],
    "cited_id": ["文献A_WOSID", "被引文献WOSID", "被引文献ID", "cited_id", "target_id"],
    "citing_title": ["文献D_题目", "文献D题目", "施引文献题目", "施引题目", "citing_title", "source_title"],
    "cited_title": ["文献A_题目", "文献A题目", "被引文献题目", "被引题目", "cited_title", "target_title"],
    "citation_text": ["引用句", "引用句文本", "引用内容", "引文内容", "引用文本", "citation_text", "context", "sentence"],
    "citation_count": ["引用次数", "被引次数", "引文次数", "citation_count", "cite_count", "count", "频次"],
}


def clean_col(x):
    return re.sub(r"[\s_\-—–:：/\\（）()\[\]【】.]+", "", unicodedata.normalize("NFKC", str(x))).casefold()


def raw(x):
    return "" if pd.isna(x) else str(x).strip()


def norm_id(x):
    s = unicodedata.normalize("NFKC", raw(x)).strip()
    s = re.sub(r"(?<=\d)\.0+$", "", s)
    return re.sub(r"[\s\-—–_.,，。:：;；/\\（）()\[\]【】]+", "", s).casefold()


def norm_title(x):
    """第一层：可审计的保守题名标准化；保留标点，仅统一常见变体。"""
    s = unicodedata.normalize("NFKC", html.unescape(raw(x))).casefold()
    trans = str.maketrans({
        "—": "-", "–": "-", "−": "-", "‐": "-", "‑": "-", "﹘": "-",
        "“": '"', "”": '"', "„": '"', "‟": '"', "＂": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "＇": "'",
    })
    s = s.translate(trans)
    # 所有Unicode空白统一为单个ASCII空格；去除不可见控制字符。
    s = "".join(" " if ch.isspace() else ch for ch in s if not unicodedata.category(ch).startswith("C"))
    return re.sub(r"\s+", " ", s).strip()


def norm_title_punctless(x):
    """第二层：在第一层基础上去除Unicode标点和空白，仍是精确键而非模糊匹配。"""
    s = norm_title(x)
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "Z")) and not ch.isspace())


def norm_sentence(x):
    s = unicodedata.normalize("NFKC", html.unescape(raw(x))).casefold().strip()
    return re.sub(r"\s+", " ", s)


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("文件不存在：%s" % path)
    if path.suffix.lower() == ".csv":
        errors = []
        for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(path, dtype=object, encoding=enc, sep=None, engine="python"), enc
            except Exception as exc:
                errors.append("%s:%s" % (enc, type(exc).__name__))
        raise ValueError("CSV读取失败：" + "; ".join(errors))
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        nonempty = [(n, x) for n, x in sheets.items() if len(x) or len(x.columns)]
        if not nonempty:
            raise ValueError("Excel无可用工作表：%s" % path)
        # 优先首个能识别引用句的sheet
        chosen = next(((n, x) for n, x in nonempty if resolve_one(x.columns, "citation_text", False)), nonempty[0])
        return chosen[1], "Excel:%s" % chosen[0]
    raise ValueError("仅支持 CSV/XLSX/XLS：%s" % path)


def candidates(columns, field):
    aliases = {clean_col(a) for a in ALIASES[field]}
    exact = [c for c in columns if clean_col(c) in aliases]
    if exact:
        return exact
    return [c for c in columns if any(len(a) >= 4 and (a in clean_col(c) or clean_col(c) in a) for a in aliases)]


def resolve_one(columns, field, strict=True):
    c = candidates(list(columns), field)
    if len(c) == 1:
        return c[0]
    if strict:
        raise ValueError("字段%s识别%s；候选=%s；实际字段=%s" % (field, "失败" if not c else "不唯一", c, list(columns)))
    return None


def mapping_diagnosis(pair_df, cite_df):
    rows = []
    for role, df, fields in [
        ("完整配对", pair_df, ["A_id", "B_id", "C_id", "A_title", "B_title", "C_title", "coupling", "similarity"]),
        ("引用内容", cite_df, ["citing_id", "cited_id", "citing_title", "cited_title", "citation_text", "citation_count"]),
    ]:
        for f in fields:
            cs = candidates(df.columns, f)
            rows.append({"source_role": role, "canonical_field": f, "candidates": " | ".join(map(str, cs)), "status": "唯一" if len(cs) == 1 else ("缺失" if not cs else "不唯一"), "all_columns": " | ".join(map(str, df.columns))})
    return pd.DataFrame(rows)


def prepare_inputs(pair_raw, cite_raw):
    pmap = {f: resolve_one(pair_raw.columns, f, f in ("coupling", "similarity")) for f in ["A_id", "B_id", "C_id", "A_title", "B_title", "C_title", "coupling", "similarity"]}
    cmap = {f: resolve_one(cite_raw.columns, f, f == "citation_text") for f in ["citing_id", "cited_id", "citing_title", "cited_title", "citation_text", "citation_count"]}
    for side in ("A", "B", "C"):
        if not pmap[side + "_id"] and not pmap[side + "_title"]:
            raise ValueError("配对文件%s侧既无ID也无题目，无法建立实体。" % side)
    if not cmap["citing_id"] and not cmap["citing_title"]:
        raise ValueError("引用内容缺少施引文献ID和题目。")
    if not cmap["cited_id"] and not cmap["cited_title"]:
        raise ValueError("引用内容缺少被引文献ID和题目。")
    p = pd.DataFrame(index=pair_raw.index)
    for f, col in pmap.items():
        p[f] = pair_raw[col] if col else ""
    # 保留基础字段供分层抽样，但避免重名
    used = {x for x in pmap.values() if x}
    for col in pair_raw.columns:
        if col not in used:
            p["基础_" + str(col)] = pair_raw[col]
    c = pd.DataFrame(index=cite_raw.index)
    for f, col in cmap.items():
        c[f] = cite_raw[col] if col else ""
    c["source_row"] = np.arange(2, len(c) + 2)
    return p, c, pmap, cmap


def id_quality_table(p, c):
    rows = []
    values = []
    for source, df, fields in [("配对", p, ["A_id", "B_id", "C_id"]), ("引用内容", c, ["citing_id", "cited_id"])]:
        for f in fields:
            for v in df[f].map(raw):
                values.append((source, f, v))
    numeric_lengths = [len(norm_id(v)) for _, _, v in values if re.fullmatch(r"\d+", norm_id(v) or "")]
    if numeric_lengths:
        counts = Counter(numeric_lengths)
        mode_len = max(sorted(counts), key=lambda z: counts[z])
        accepted_lengths = {x for x, n in counts.items() if n >= max(2, int(0.05 * len(numeric_lengths)))} | {mode_len}
    else:
        mode_len, accepted_lengths = None, set()
    for source, field, v in values:
        n = norm_id(v)
        missing = not n
        pure = bool(re.fullmatch(r"\d+", n or ""))
        length_bad = pure and bool(accepted_lengths) and len(n) not in accepted_lengths
        rows.append({"source": source, "field": field, "raw_id": v, "normalized_id": n, "missing": missing, "non_pure_numeric": bool(n) and not pure, "length": len(n) if n else np.nan, "length_abnormal": length_bad, "id_usable": pure and not length_bad, "modal_numeric_length": mode_len})
    detail = pd.DataFrame(rows).drop_duplicates(["source", "field", "raw_id"])
    return detail, accepted_lengths


def usable_id(v, accepted_lengths):
    n = norm_id(v)
    return n if re.fullmatch(r"\d+", n or "") and (not accepted_lengths or len(n) in accepted_lengths) else ""


def collision_diagnosis(p, accepted_lengths):
    entities = []
    for side in ("A", "B", "C"):
        for r in p.itertuples():
            entities.append({"side": side, "id": usable_id(getattr(r, side + "_id"), accepted_lengths), "title": norm_title(getattr(r, side + "_title")), "raw_id": raw(getattr(r, side + "_id")), "raw_title": raw(getattr(r, side + "_title"))})
    e = pd.DataFrame(entities).drop_duplicates()
    rows = []
    for key, g in e[e.id.ne("")].groupby("id"):
        titles = sorted(set(g.title) - {""})
        if len(titles) > 1:
            rows.append({"diagnostic_type": "同ID多题名", "key": key, "distinct_count": len(titles), "values": " | ".join(titles)})
    for key, g in e[e.title.ne("")].groupby("title"):
        ids = sorted(set(g.id) - {""})
        if len(ids) > 1:
            rows.append({"diagnostic_type": "同题名多ID", "key": key, "distinct_count": len(ids), "values": " | ".join(ids)})
    return pd.DataFrame(rows, columns=["diagnostic_type", "key", "distinct_count", "values"]), e


def build_registry(triples, accepted_lengths, progress=None):
    """以(施引D题名,被引A题名)为正式pair主键；WOSID仅建立审计索引。"""
    recs = []
    rows = triples.iterrows()
    if progress:
        rows = progress.iter(rows, "建立题名文献对", total=len(triples))
    for ti, r in rows:
        for side in ("B", "C"):
            cited_title, citing_title = raw(r.A_title), raw(r[side + "_title"])
            recs.append({
                "triple_index": ti, "side": side,
                "cited_id_raw": raw(r.A_id), "citing_id_raw": raw(r[side + "_id"]),
                "cited_title_raw": cited_title, "citing_title_raw": citing_title,
                "cited_title_key": norm_title(cited_title), "citing_title_key": norm_title(citing_title),
                "cited_title_key_punctless": norm_title_punctless(cited_title),
                "citing_title_key_punctless": norm_title_punctless(citing_title),
            })
    source = pd.DataFrame(recs)
    # 第一层保守题名键相同即为同一文献对；ID差异不拆分pair。
    valid = source.cited_title_key.ne("") & source.citing_title_key.ne("")
    unique_keys = source.loc[valid, ["citing_title_key", "cited_title_key"]].drop_duplicates().reset_index(drop=True)
    unique_keys.insert(0, "pair_id", np.arange(len(unique_keys)))
    key_to_pid = {(r.citing_title_key, r.cited_title_key): r.pair_id for r in unique_keys.itertuples()}
    source["pair_id"] = [key_to_pid.get((r.citing_title_key, r.cited_title_key), np.nan) for r in source.itertuples()]

    pair_rows = []
    for key, g in source[valid].groupby(["citing_title_key", "cited_title_key"], sort=False):
        pair_rows.append({
            "pair_id": key_to_pid[key], "citing_title_key": key[0], "cited_title_key": key[1],
            "citing_title_key_punctless": g.citing_title_key_punctless.iloc[0],
            "cited_title_key_punctless": g.cited_title_key_punctless.iloc[0],
            "citing_title_raw": " | ".join(sorted(set(g.citing_title_raw) - {""})),
            "cited_title_raw": " | ".join(sorted(set(g.cited_title_raw) - {""})),
            "citing_id_raw": " | ".join(sorted(set(g.citing_id_raw) - {""})),
            "cited_id_raw": " | ".join(sorted(set(g.cited_id_raw) - {""})),
        })
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        pairs = pd.DataFrame(columns=["pair_id", "citing_title_key", "cited_title_key", "citing_title_key_punctless", "cited_title_key_punctless", "citing_title_raw", "cited_title_raw", "citing_id_raw", "cited_id_raw"])

    title_lookup_1, title_lookup_2, id_lookup = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in pairs.itertuples():
        title_lookup_1[(r.citing_title_key, r.cited_title_key)].add(r.pair_id)
        if r.citing_title_key_punctless and r.cited_title_key_punctless:
            title_lookup_2[(r.citing_title_key_punctless, r.cited_title_key_punctless)].add(r.pair_id)
    # ID仅用于核对它通常指向哪个题名pair，绝不用于正式纳入。
    for r in source[valid].itertuples():
        ci, ai = usable_id(r.citing_id_raw, accepted_lengths), usable_id(r.cited_id_raw, accepted_lengths)
        if ci and ai and pd.notna(r.pair_id):
            id_lookup[(ci, ai)].add(int(r.pair_id))
    membership = source[["triple_index", "side", "pair_id"]].copy()
    return pairs, membership, title_lookup_1, title_lookup_2, id_lookup


def match_citation(row, title_lookup_1, title_lookup_2, id_lookup, accepted_lengths):
    """题名唯一决定正式匹配；WOSID矛盾只标记审计冲突，不推翻题名结果。"""
    t1 = (norm_title(row.citing_title), norm_title(row.cited_title))
    t2 = (norm_title_punctless(row.citing_title), norm_title_punctless(row.cited_title))
    if not all(t1):
        return np.nan, "不可匹配", "施引或被引题名缺失", False, "题名缺失，不回退WOSID"
    hits = title_lookup_1.get(t1, set())
    level = "题名保守精确"
    if not hits:
        hits = title_lookup_2.get(t2, set()) if all(t2) else set()
        level = "题名去标点精确"
    if len(hits) > 1:
        return np.nan, "不可匹配", "标准化题名对对应多个对象", False, ""
    if not hits:
        return np.nan, "未匹配", "题名对未找到", False, ""
    pair_id = next(iter(hits))
    ikey = (usable_id(row.citing_id, accepted_lengths), usable_id(row.cited_id, accepted_lengths))
    ih = id_lookup.get(ikey, set()) if all(ikey) else set()
    conflict = bool(ih and pair_id not in ih)
    detail = "WOSID指向pair=%s，题名指向pair=%s；保留题名结果" % ("|".join(map(str, sorted(ih))), pair_id) if conflict else ""
    return pair_id, level, "匹配成功（WOSID冲突但保留题名）" if conflict else "匹配成功", conflict, detail


def fuzzy_candidates(unmatched, pairs, progress=None):
    out = []
    rows = unmatched.head(300).itertuples()
    if progress:
        rows = progress.iter(rows, "生成模糊复核候选", total=min(len(unmatched), 300))
    for r in rows:
        ct, at = norm_title(r.citing_title), norm_title(r.cited_title)
        if not ct or not at:
            continue
        scored = []
        for p in pairs.itertuples():
            if not p.citing_title_key or not p.cited_title_key:
                continue
            score = (SequenceMatcher(None, ct, p.citing_title_key).ratio() + SequenceMatcher(None, at, p.cited_title_key).ratio()) / 2
            if score >= FUZZY_CUT:
                scored.append((score, p))
        for score, p in sorted(scored, key=lambda z: z[0], reverse=True)[:3]:
            out.append({"source_row": r.source_row, "input_citing_title": r.citing_title, "input_cited_title": r.cited_title, "candidate_pair_id": p.pair_id, "candidate_citing_title": p.citing_title_raw, "candidate_cited_title": p.cited_title_raw, "similarity": round(score, 4), "warning": "仅供人工复核，未纳入正式结果"})
    return pd.DataFrame(out)


def count_audit(group):
    vals = pd.to_numeric(group.citation_count, errors="coerce").dropna().tolist()
    if not vals:
        return pd.Series({"reported_count_values": "", "reported_count": np.nan, "reported_count_min": np.nan, "reported_count_max": np.nan, "reported_count_consistent": np.nan})
    cnt = Counter(vals); m = max(cnt.values()); mode = min(x for x, n in cnt.items() if n == m)
    return pd.Series({"reported_count_values": " | ".join("%g" % x for x in sorted(cnt)), "reported_count": mode, "reported_count_min": min(vals), "reported_count_max": max(vals), "reported_count_consistent": len(cnt) == 1})


def assign_group(sim, coupling):
    if pd.isna(sim) or pd.isna(coupling):
        return "阈值缺失"
    return ("Q1" if sim >= SIM_CUT else "Q0") + ("1" if coupling >= COUPLING_CUT else "0")


def add_metrics(t):
    obs = t.AB_observable & t.AC_observable
    t["bilateral_observable"] = obs
    t["I_min"] = np.where(obs, t[["AB_strength", "AC_strength"]].min(axis=1), np.nan)
    t["I_mean"] = np.where(obs, t[["AB_strength", "AC_strength"]].mean(axis=1), np.nan)
    t["I_total"] = np.where(obs, t.AB_strength + t.AC_strength, np.nan)
    t["absolute_gap"] = np.where(obs, (t.AB_strength - t.AC_strength).abs(), np.nan)
    t["D_norm"] = np.where(obs & t.I_total.gt(0), t.absolute_gap / t.I_total, np.nan)
    t["S_intensity"] = np.where(obs & t.I_total.gt(0), 2 * t.I_min / t.I_total, np.nan)
    t["complete_equal"] = np.where(obs, t.AB_strength.eq(t.AC_strength), np.nan)
    t["gap_le_1"] = np.where(obs, t.absolute_gap.le(1), np.nan)
    t["bilateral_ge_2"] = np.where(obs, t.AB_strength.ge(2) & t.AC_strength.ge(2), np.nan)
    t["H3_combo"] = np.where(~obs, "不可观测", np.where(t.AB_strength.ge(2) & t.AC_strength.ge(2), "双侧>=2", np.where(t.AB_strength.ge(2), "仅B>=2", np.where(t.AC_strength.ge(2), "仅C>=2", "双侧<2"))))
    return t


def comparable_sample(primary, strata_cols):
    base = primary[primary.group.eq("Q11")].copy()
    base["sample_role"] = "Q11基准组"
    n = len(base); pieces = [base]
    audit = [{"group": "Q11", "available": n, "selected": n, "method": "全量基准组", "seed": SEED}]
    for group in ("Q10", "Q01", "Q00"):
        pool = primary[primary.group.eq(group)].copy()
        target = min(n, len(pool))
        method = "全量（该组小于Q11）" if len(pool) < n else "固定种子随机抽样"
        chosen = pd.DataFrame(columns=pool.columns)
        usable = [c for c in strata_cols if c in pool and pool[c].notna().any() and base[c].notna().any()]
        if target and len(pool) > target and usable:
            # 按Q11联合分层分布分配，余量固定种子补齐；不使用任何结果变量。
            method = "按%s分层抽样，余量固定种子补齐" % ",".join(usable)
            quotas = base.groupby(usable, dropna=False).size() / max(len(base), 1) * target
            selected = []
            for key, q in quotas.items():
                key = key if isinstance(key, tuple) else (key,)
                mask = pd.Series(True, index=pool.index)
                for col, val in zip(usable, key):
                    mask &= pool[col].isna() if pd.isna(val) else pool[col].eq(val)
                k = min(int(math.floor(q)), int(mask.sum()))
                if k:
                    selected.extend(pool[mask].sample(k, random_state=SEED).index.tolist())
            chosen = pool.loc[sorted(set(selected))]
            if len(chosen) < target:
                extra = pool.drop(chosen.index).sample(target - len(chosen), random_state=SEED)
                chosen = pd.concat([chosen, extra])
        elif target:
            chosen = pool if len(pool) <= target else pool.sample(target, random_state=SEED)
        chosen = chosen.copy(); chosen["sample_role"] = "可比控制组"
        pieces.append(chosen)
        audit.append({"group": group, "available": len(pool), "selected": len(chosen), "method": method, "seed": SEED})
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(audit)


def descriptives(df, sample_name):
    rows = []
    for g in ("Q11", "Q10", "Q01", "Q00"):
        z = df[df.group.eq(g)]
        for metric in ("I_min", "I_mean", "I_total", "D_norm", "S_intensity"):
            x = pd.to_numeric(z[metric], errors="coerce").dropna()
            rows.append({"sample": sample_name, "group": g, "metric": metric, "n": len(x), "mean": x.mean(), "std": x.std(ddof=1), "median": x.median(), "p25": x.quantile(.25), "p75": x.quantile(.75), "min": x.min(), "max": x.max()})
        for metric in ("complete_equal", "gap_le_1", "bilateral_ge_2"):
            x = pd.to_numeric(z[metric], errors="coerce").dropna()
            rows.append({"sample": sample_name, "group": g, "metric": metric, "n": len(x), "mean": x.mean(), "count_true": int(x.sum())})
    return pd.DataFrame(rows)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y): return np.nan
    return float(sum(np.sum(v > y) - np.sum(v < y) for v in x) / (len(x) * len(y)))


def inferential(df, sample_name):
    try:
        from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
    except ImportError:
        return pd.DataFrame([{"sample": sample_name, "error": "未安装scipy；请执行pip install -r requirements.txt"}]), pd.DataFrame()
    continuous, proportions = [], []
    q = df[df.group.eq("Q11")]
    for control in ("Q10", "Q01", "Q00"):
        c = df[df.group.eq(control)]
        for metric in ("I_min", "I_mean", "I_total", "D_norm", "S_intensity"):
            x = pd.to_numeric(q[metric], errors="coerce").dropna(); y = pd.to_numeric(c[metric], errors="coerce").dropna()
            stat = pval = np.nan
            if len(x) and len(y): stat, pval = mannwhitneyu(x, y, alternative="two-sided")
            continuous.append({"sample": sample_name, "comparison": "Q11 vs " + control, "metric": metric, "n_Q11": len(x), "n_control": len(y), "U": stat, "p_value": pval, "cliffs_delta_Q11_minus_control": cliffs_delta(x, y)})
        for metric in ("complete_equal", "gap_le_1", "bilateral_ge_2"):
            x = pd.to_numeric(q[metric], errors="coerce").dropna().astype(int); y = pd.to_numeric(c[metric], errors="coerce").dropna().astype(int)
            a, b, cc, d = int(x.sum()), int(len(x)-x.sum()), int(y.sum()), int(len(y)-y.sum())
            table = np.array([[a,b],[cc,d]])
            test = "Fisher" if (table < 5).any() else "Chi-square"
            if len(x) and len(y):
                if test == "Fisher": stat, pval = fisher_exact(table)
                else: stat, pval = chi2_contingency(table, correction=False)[:2]
            else: stat = pval = np.nan
            p1 = a/len(x) if len(x) else np.nan; p0 = cc/len(y) if len(y) else np.nan
            diff = p1-p0 if len(x) and len(y) else np.nan
            se = math.sqrt(p1*(1-p1)/len(x)+p0*(1-p0)/len(y)) if len(x) and len(y) else np.nan
            aa,bb,ccc,dd = a+.5,b+.5,cc+.5,d+.5
            odds = aa*dd/(bb*ccc); selog = math.sqrt(1/aa+1/bb+1/ccc+1/dd)
            proportions.append({"sample": sample_name, "comparison": "Q11 vs " + control, "metric": metric, "n_Q11": len(x), "n_control": len(y), "rate_Q11": p1, "rate_control": p0, "proportion_difference": diff, "diff_ci95_low": diff-1.96*se if len(x) and len(y) else np.nan, "diff_ci95_high": diff+1.96*se if len(x) and len(y) else np.nan, "odds_ratio": odds, "or_ci95_low": math.exp(math.log(odds)-1.96*selog), "or_ci95_high": math.exp(math.log(odds)+1.96*selog), "test": test, "statistic_or_OR": stat, "p_value": pval})
        # H3四组合总体分布比较
        cats = ["双侧>=2", "仅B>=2", "仅C>=2", "双侧<2"]
        table = np.array([[int((q.H3_combo == k).sum()) for k in cats], [int((c.H3_combo == k).sum()) for k in cats]])
        keep = table.sum(axis=0) > 0
        if keep.sum() >= 2 and table.sum(axis=1).min() > 0:
            stat, pval = chi2_contingency(table[:, keep], correction=False)[:2]
        else: stat = pval = np.nan
        proportions.append({"sample": sample_name, "comparison": "Q11 vs " + control, "metric": "H3四种组合", "n_Q11": int(table[0].sum()), "n_control": int(table[1].sum()), "test": "Chi-square", "statistic_or_OR": stat, "p_value": pval})
    return pd.DataFrame(continuous), pd.DataFrame(proportions)


def make_charts(primary, comparable, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = ["Q11", "Q10", "Q01", "Q00"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric in zip(axes, ["I_total", "D_norm", "S_intensity"]):
        vals = [pd.to_numeric(comparable.loc[comparable.group.eq(g), metric], errors="coerce").dropna() for g in order]
        ax.boxplot(vals, labels=order, showfliers=False); ax.set_title("可比样本 " + metric); ax.set_xlabel("组别")
    fig.tight_layout(); fig.savefig(outdir / "四组核心指标_可比样本.png", dpi=180); plt.close(fig)
    cov = primary.groupby("group").bilateral_observable.agg(["mean", "size"]).reindex(order)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(order, cov["mean"].fillna(0)); ax.set_ylim(0,1); ax.set_ylabel("双侧强度可观测率"); ax.set_title("四组引用内容覆盖率")
    for i, v in enumerate(cov["mean"]):
        if pd.notna(v): ax.text(i, v+.02, "%.1f%%\n(n=%d)" % (100*v, cov.iloc[i]["size"]), ha="center")
    fig.tight_layout(); fig.savefig(outdir / "四组覆盖率.png", dpi=180); plt.close(fig)


def safe_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


EXCEL_MAX_DATA_ROWS = 100000  # Excel仅承载便于查看的预览；完整数据始终保存在同名CSV。


def excel_preview(df, sheet_label):
    """避免超过Excel单Sheet上限；大表仅写前N行，并返回内容清单信息。"""
    total_rows = len(df)
    if total_rows > EXCEL_MAX_DATA_ROWS:
        shown = df.head(EXCEL_MAX_DATA_ROWS).copy()
        mode = "预览前%d行；完整数据见对应CSV" % EXCEL_MAX_DATA_ROWS
    else:
        shown = df
        mode = "完整写入"
    info = {
        "工作表": sheet_label,
        "原始总行数": total_rows,
        "Excel写入行数": len(shown),
        "写入方式": mode,
    }
    return shown, info


def run(pair_file=DEFAULT_PAIR, citation_file=DEFAULT_CITATION, output_dir=None, progress=None):
    pair_file, citation_file = Path(pair_file), Path(citation_file)
    output_dir = Path(output_dir) if output_dir else pair_file.parent / "ITC四组引用强度分析输出"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or ProgressReporter()
    progress.stage(2, "扫描文件")
    for input_path in (pair_file, citation_file):
        if not input_path.exists():
            raise FileNotFoundError("文件不存在：%s" % input_path)
    progress.stage(7, "读取完整配对CSV")
    with progress.heartbeat("正在读取完整配对CSV"):
        pair_raw, penc = read_table(pair_file)
    progress.stage(14, "读取引用内容")
    with progress.heartbeat("正在读取引用内容"):
        cite_raw, cenc = read_table(citation_file)
    progress.stage(22, "字段识别")
    pair_raw.columns = [str(x).strip() for x in pair_raw.columns]; cite_raw.columns = [str(x).strip() for x in cite_raw.columns]
    mapping = mapping_diagnosis(pair_raw, cite_raw)
    safe_csv(mapping, output_dir / "字段识别诊断.csv")
    p, c, pmap, cmap = prepare_inputs(pair_raw, cite_raw)
    progress.stage(30, "题名标准化")
    with progress.heartbeat("正在执行题名标准化与质量诊断"):
        quality, accepted_lengths = id_quality_table(p, c)
        collisions, entities = collision_diagnosis(p, accepted_lengths)
        pairs, membership, title_lookup_1, title_lookup_2, id_lookup = build_registry(p, accepted_lengths, progress)
    progress.stage(38, "四组划分")
    with progress.heartbeat("正在进行四组划分"):
        p["coupling"] = pd.to_numeric(p.coupling, errors="coerce"); p["similarity"] = pd.to_numeric(p.similarity, errors="coerce")
        p["group"] = [assign_group(s, x) for s, x in zip(p.similarity, p.coupling)]
        p.insert(0, "triple_id", np.arange(1, len(p)+1))
    progress.stage(47, "文献对聚合")
    cite_rows = progress.iter(c.itertuples(index=False), "匹配引用内容", total=len(c))
    matches = [match_citation(r, title_lookup_1, title_lookup_2, id_lookup, accepted_lengths) for r in cite_rows]
    c["pair_id"] = [x[0] for x in matches]
    c["match_level"] = [x[1] for x in matches]
    c["match_reason"] = [x[2] for x in matches]
    c["wosid_conflict"] = [x[3] for x in matches]
    c["wosid_conflict_detail"] = [x[4] for x in matches]
    c["citation_text_norm"] = c.citation_text.map(norm_sentence)
    matched = c[c.pair_id.notna()].copy(); matched["pair_id"] = matched.pair_id.astype(int)
    with_text = matched[matched.citation_text_norm.ne("")]
    dedup = with_text.drop_duplicates(["pair_id", "citation_text_norm"])
    strength = dedup.groupby("pair_id").citation_text_norm.nunique()
    audits = matched.groupby("pair_id")[["citation_count"]].apply(count_audit) if len(matched) else pd.DataFrame()
    pair_stats = pairs.copy(); pair_stats["observable"] = pair_stats.pair_id.isin(strength.index); pair_stats["unique_strength"] = pair_stats.pair_id.map(strength).astype(float)
    if len(audits): pair_stats = pair_stats.merge(audits.reset_index(), on="pair_id", how="left")
    else:
        for col in ["reported_count_values","reported_count","reported_count_min","reported_count_max","reported_count_consistent"]: pair_stats[col] = np.nan
    pair_stats["strength_minus_reported"] = pair_stats.unique_strength - pair_stats.reported_count
    member_summary = membership.groupby("pair_id").agg(triple_membership_count=("triple_index","nunique"), sides_used=("side",lambda x:" | ".join(sorted(set(x))))).reset_index()
    pair_stats = pair_stats.merge(member_summary, on="pair_id", how="left")
    progress.stage(57, "三元组关联")
    # 回连所有三元组；不存在引用覆盖时保留NaN，绝不填0。
    with progress.heartbeat("正在关联三元组并计算指标"):
        t = p.copy()
    for side in ("B", "C"):
        mm = membership[membership.side.eq(side)].set_index("triple_index").pair_id
        t["A%s_pair_id" % side] = t.index.map(mm)
        stat = pair_stats.set_index("pair_id")
        t["A%s_observable" % side] = t["A%s_pair_id" % side].map(stat.observable).fillna(False)
        t["A%s_strength" % side] = t["A%s_pair_id" % side].map(stat.unique_strength)
        t["A%s_reported_count" % side] = t["A%s_pair_id" % side].map(stat.reported_count)
    t = add_metrics(t)
    primary = t[t.bilateral_observable & t.group.isin(["Q11","Q10","Q01","Q00"])].copy()
    # 仅基础协变量；禁止使用I_total及任何结果变量。优先显式A分组/学科/年份。
    baseline = [x for x in t.columns if x.startswith("基础_") and any(k in clean_col(x) for k in ("a分组","学科","领域","年份","year","category"))]
    comparable, sample_audit = comparable_sample(primary, baseline)
    coverage = t[t.group.isin(["Q11","Q10","Q01","Q00"])].groupby("group").agg(triples=("triple_id","size"), bilateral_observable=("bilateral_observable","sum"), coverage_rate=("bilateral_observable","mean"), AB_observable_rate=("AB_observable","mean"), AC_observable_rate=("AC_observable","mean")).reset_index()
    progress.stage(68, "统计检验")
    with progress.heartbeat("正在执行描述统计与统计检验"):
        desc = pd.concat([descriptives(primary, "全量可观测主分析"), descriptives(comparable, "1:1可比样本")], ignore_index=True)
        mw1, prop1 = inferential(primary, "全量可观测主分析"); mw2, prop2 = inferential(comparable, "1:1可比样本")
        mw = pd.concat([mw1,mw2], ignore_index=True); prop = pd.concat([prop1,prop2], ignore_index=True)
    unmatched = c[c.pair_id.isna()].copy(); fuzzy = fuzzy_candidates(unmatched, pairs, progress)
    wosid_conflicts = c[c.wosid_conflict].copy()
    match_summary = c.groupby(["match_level","match_reason"], dropna=False).size().rename("rows").reset_index()
    dedup_out = dedup.merge(pairs, on="pair_id", how="left")
    progress.stage(78, "写出CSV")
    outputs = {
        "四组全量三元组.csv": t, "主分析_双侧可观测.csv": primary, "可比样本_1比1.csv": comparable, "文献对引用强度.csv": pair_stats,
        "引用句去重明细.csv": dedup_out, "引用次数审计.csv": pair_stats, "覆盖率.csv": coverage, "描述统计.csv": desc,
        "MannWhitneyU与CliffsDelta.csv": mw, "比例检验与效应量.csv": prop, "抽样审计.csv": sample_audit,
        "ID质量检查.csv": quality, "ID题名关联诊断.csv": collisions, "WOSID与题名冲突明细.csv": wosid_conflicts, "未匹配明细.csv": unmatched, "模糊题名候选_仅人工复核.csv": fuzzy,
    }
    for name, df in progress.iter(outputs.items(), "写出CSV", total=len(outputs)):
        safe_csv(df, output_dir / name)
    progress.stage(86, "写出Excel")
    notes = pd.DataFrame([
        {"item":"主分析口径","value":"仅双侧强度可观测三元组；配对存在但引用内容无有效句为不可观测(NaN)，不记0。"},
        {"item":"Excel大表口径","value":"Excel单Sheet最多1048576行。超过10万行的明细在Excel中仅保留前10万行预览，完整数据保存在同目录对应CSV；详见Excel内容清单。"},
        {"item":"分组阈值","value":"Q11:sim>=0.60且coupling>=10；Q10/Q01/Q00依此类推。"},
        {"item":"匹配主键","value":"正式匹配仅以(施引D题名,被引A题名)为主键：先保守标准化精确，再去标点精确。WOSID仅审计，不参与纳入。"},
        {"item":"WOSID冲突","value":"题名匹配成功时即保留；若WOSID指向其他pair，仅标记冲突并输出明细，不排除。题名缺失不回退WOSID。模糊题名仅候选。"},
        {"item":"可比样本限制","value":"优先按可识别基础字段分层；不可用时固定种子随机抽样。未按I_total或其他结果变量匹配。"},
        {"item":"输入编码","value":"配对=%s；引用=%s" % (penc,cenc)},
    ])
    report_path = output_dir / "ITC四组引用强度分析报告.xlsx"
    sheets = {"分析说明":notes,"全量三元组":t,"主分析_可观测":primary,"可比样本":comparable,"覆盖率":coverage,"描述统计":desc,"MWU与Cliff":mw,"比例检验效应量":prop,"H3四组合":pd.crosstab(primary.group,primary.H3_combo).reset_index(),"抽样审计":sample_audit,"文献对强度":pair_stats,"匹配汇总":match_summary,"ID质量":quality,"ID题名关联诊断":collisions,"WOSID题名冲突":wosid_conflicts,"未匹配":unmatched,"模糊候选不入结果":fuzzy,"字段识别":mapping,"去重引用句":dedup_out}
    prepared_sheets = []
    inventory_rows = []
    for name, df in sheets.items():
        shown, info = excel_preview(df, name)
        prepared_sheets.append((name, shown))
        inventory_rows.append(info)
    inventory = pd.DataFrame(inventory_rows)
    with pd.ExcelWriter(report_path, engine="xlsxwriter") as w:
        inventory.to_excel(w, sheet_name="Excel内容清单", index=False)
        for name, df in progress.iter(prepared_sheets, "写入Excel工作表", total=len(prepared_sheets)):
            df.to_excel(w, sheet_name=name[:31], index=False)
    progress.stage(96, "生成图表")
    with progress.heartbeat("正在生成图表"):
        make_charts(primary, comparable, output_dir)
    progress.stage(100, "完成")
    logging.info("完成：%s", output_dir)
    return {"triples":t,"primary":primary,"comparable":comparable,"pairs":pair_stats,"coverage":coverage,"unmatched":unmatched,"fuzzy":fuzzy,"quality":quality,"collisions":collisions,"wosid_conflicts":wosid_conflicts,"sample_audit":sample_audit,"mw":mw,"proportions":prop}


def main():
    ap = argparse.ArgumentParser(description="ITC+非ITC四组引用强度一键分析")
    ap.add_argument("--pair-file", default=str(DEFAULT_PAIR)); ap.add_argument("--citation-file", default=str(DEFAULT_CITATION)); ap.add_argument("--output-dir", default="")
    args = ap.parse_args(); out = Path(args.output_dir).expanduser() if args.output_dir else Path(args.pair_file).expanduser().parent / "ITC四组引用强度分析输出"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(out/"运行日志.log",encoding="utf-8"),logging.StreamHandler(sys.stdout)])
    reporter = ProgressReporter()
    reporter.stage(0, "正在运行：初始化")
    try:
        result = run(Path(args.pair_file).expanduser(), Path(args.citation_file).expanduser(), out, progress=reporter)
        print("运行成功：全量%d，主分析可观测%d，可比样本%d；输出：%s" % (len(result["triples"]),len(result["primary"]),len(result["comparable"]),out))
    except Exception as exc:
        logging.exception("运行失败，停止阶段：%s", reporter.stage_name)
        print("运行失败：停止在【%s】；原因：%s；日志：%s" % (reporter.stage_name, exc, out/"运行日志.log"))
        raise SystemExit(1)


if __name__ == "__main__": main()
