import numpy as np
import pandas as pd

import config
from core import enrich, log, stable_hash
from io_utils import iter_raw_files


def _rank(x):
    x = x.copy()
    x["_hash"] = x.record_key.map(stable_hash)
    return x.sort_values(["_hash", "record_key"])


def _stratified_candidates(x, exact_set, context_set, a_section_set, a_set, section_set, fallback_n):
    """为大象限保留分层候选和确定性哈希回退池。"""
    exact = x[[v in exact_set for v in zip(x.a_key, x.context_key)]]
    same_context = x[x.context_key.isin(context_set)]
    same_a_section = x[[v in a_section_set for v in zip(x.a_key, x.section_combo)]]
    same_a = x[x.a_key.isin(a_set)]
    same_section = x[x.section_combo.isin(section_set)]
    selections = [
        exact.groupby(["quadrant", "a_key", "context_key"], dropna=False, sort=False).head(config.POOL_PER_STRATUM),
        same_context.groupby(["quadrant", "context_key"], dropna=False, sort=False).head(config.POOL_PER_STRATUM),
        same_a_section.groupby(["quadrant", "a_key", "section_combo"], dropna=False, sort=False).head(config.POOL_PER_STRATUM),
        same_a.groupby(["quadrant", "a_key"], dropna=False, sort=False).head(config.POOL_PER_STRATUM),
        same_section.groupby(["quadrant", "section_combo"], dropna=False, sort=False).head(config.POOL_PER_STRATUM),
        x.groupby("quadrant", sort=False).head(fallback_n),
    ]
    return pd.concat(selections, ignore_index=False).drop_duplicates("record_key")


def build_candidate_pool(target, raw_paths):
    """Q01 全量保留；Q10/Q00 使用分层及哈希池控制内存，均以记录键去重。"""
    a_set, section_set, context_set = set(target.a_key), set(target.section_combo), set(target.context_key)
    exact_set = set(zip(target.a_key, target.context_key))
    a_section_set = set(zip(target.a_key, target.section_combo))
    kept = []
    q01_raw_rows = 0
    per_block_fallback = max(500, config.GLOBAL_POOL_PER_QUADRANT // 10)
    for path, block, raw in iter_raw_files(raw_paths):
        x = _rank(enrich(raw[config.REQUIRED_COLUMNS]))
        q01_raw_rows += int(x.quadrant.eq("Q01").sum())
        x = x.drop_duplicates("record_key")
        # Q01 仅 21,442 行，完整纳入，不能因三元组相同丢掉不同引用记录。
        q01 = x[x.quadrant.eq("Q01")]
        large = x[x.quadrant.isin(["Q10", "Q00"])]
        sampled = _stratified_candidates(
            large, exact_set, context_set, a_section_set, a_set, section_set, per_block_fallback
        ) if not large.empty else large
        part = pd.concat([q01, sampled], ignore_index=False).drop_duplicates("record_key")
        kept.append(part)
        log(f"候选池 {path.name} 第 {block} 块：保留 {len(part):,} 条（Q01 {len(q01):,}）")
    if not kept:
        return pd.DataFrame()

    pool = _rank(pd.concat(kept, ignore_index=True).drop_duplicates("record_key"))
    q01 = pool[pool.quadrant.eq("Q01")]
    large = pool[pool.quadrant.isin(["Q10", "Q00"])]
    final_large = _stratified_candidates(
        large, exact_set, context_set, a_section_set, a_set, section_set,
        max(config.GLOBAL_POOL_PER_QUADRANT, len(target)),
    ) if not large.empty else large
    final = pd.concat([q01, final_large], ignore_index=True).drop_duplicates("record_key")
    final = _rank(final).reset_index(drop=True)
    final.attrs["q01_raw_rows"] = q01_raw_rows
    return final


def distance(anchor, candidates):
    return (candidates["sbert相似度"].sub(anchor["sbert相似度"]).abs()
            + np.log1p(candidates["耦合强度"]).sub(np.log1p(anchor["耦合强度"])).abs())


def match_controls(target, pool):
    rows, used = [], {q: set() for q in ("Q10", "Q01", "Q00")}
    # 不使用情感；逐层放宽 A、Section 组合和 Classification 组合。
    levels = [
        (1, lambda a, p: p[(p.a_key == a.a_key) & (p.section_combo == a.section_combo) & (p.classification_combo == a.classification_combo)]),
        (2, lambda a, p: p[(p.section_combo == a.section_combo) & (p.classification_combo == a.classification_combo)]),
        (3, lambda a, p: p[(p.a_key == a.a_key) & (p.section_combo == a.section_combo)]),
        (4, lambda a, p: p[p.a_key == a.a_key]),
        (5, lambda a, p: p[p.section_combo == a.section_combo]),
        (6, lambda a, p: p),
    ]
    for i, anchor in enumerate(target.itertuples(index=False), 1):
        a = pd.Series(anchor._asdict())
        group_id = f"M{i:06d}"
        for q in ("Q10", "Q01", "Q00"):
            base = pool[(pool.quadrant == q) & ~pool.record_key.isin(used[q])]
            chosen = None
            for level, selector in levels:
                available = selector(a, base)
                if not available.empty:
                    available = available.assign(_distance=distance(a, available)).sort_values(["_distance", "_hash"])
                    chosen = available.iloc[0].copy()
                    chosen["match_level"] = level
                    break
            if chosen is not None:
                used[q].add(chosen.record_key)
                d = chosen.to_dict()
                d.update({"match_group_id": group_id, "anchor_pair_key": a.pair_key,
                          "anchor_record_key": a.record_key, "distance": float(chosen._distance), "control_rank": 1})
                rows.append(d)
        if i % 500 == 0:
            log(f"已匹配 {i:,}/{len(target):,} 条 Q11")
    return pd.DataFrame(rows)


def run_matching(itc, raw_paths):
    original_itc_n = len(itc)
    target = _rank(itc).drop_duplicates("record_key").reset_index(drop=True)
    exact_duplicates = original_itc_n - len(target)
    if exact_duplicates:
        log(f"ITC 完全重复记录已删除：{exact_duplicates:,} 条")
    target["match_group_id"] = [f"M{i:06d}" for i in range(1, len(target) + 1)]
    target["anchor_pair_key"] = target.pair_key
    target["anchor_record_key"] = target.record_key
    target["match_level"], target["distance"], target["control_rank"] = 0, 0.0, 0
    pool = build_candidate_pool(target, raw_paths)
    controls = match_controls(target, pool)
    sample = pd.concat([target, controls], ignore_index=True, sort=False).drop(columns=["_hash", "_distance"], errors="ignore")
    out = config.OUTPUT_DIR / "02_匹配样本.csv.gz"
    sample.to_csv(out, index=False, encoding="utf-8-sig", compression="gzip")
    counts = sample.quadrant.value_counts().to_dict()
    q01_pool = pool[pool.quadrant.eq("Q01")]
    q01_success = counts.get("Q01", 0)
    q01_raw_rows = pool.attrs.get("q01_raw_rows", config.EXPECTED_COUNTS["Q01"])
    log(f"Q01匹配诊断：原始Q01行数={q01_raw_rows:,}，唯一record数={q01_pool.record_key.nunique():,}，候选池数={len(q01_pool):,}，成功匹配数={q01_success:,}")
    if q01_success < len(target):
        shortage = len(target) - q01_success
        log(f"Q01未满配原因：可用且不重复的record_key候选不足（缺少 {shortage:,} 条）；同一pair_key下不同record_key已允许分别匹配。")
    log(f"1:1 匹配完成：{counts}")
    return sample
