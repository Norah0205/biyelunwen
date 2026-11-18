"""汇总真实测试关键指标并执行关系型结果完整性校验。"""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).parent / "_real_test_output"
book = pd.read_excel(root / "引用强度分析报告.xlsx", sheet_name=None)
summary = book["匹配汇总"].iloc[0]
desc = book["H1-H3描述指标"]
pairs = pd.read_csv(root / "文献对引用强度.csv", dtype={"citing_D": str, "cited_A": str})
triples = pd.read_csv(root / "三元组引用强度.csv", dtype={"A": str, "B": str, "C": str})
unmatched = pd.read_csv(root / "未匹配明细.csv")

# 强制校验H3定义及pair左连接一致性。
assert (triples["H3_bilateral_sustained_use"] == ((triples.AB_strength >= 2) & (triples.AC_strength >= 2))).all()
pidx = pairs.set_index("pair_id")["unique_strength"]
assert (triples.AB_pair_id.map(pidx).fillna(0).astype(int) == triples.AB_strength).all()
assert (triples.AC_pair_id.map(pidx).fillna(0).astype(int) == triples.AC_strength).all()

reported = pairs.reported_count.notna()
consistent = reported & pairs.strength_count_consistency.eq("一致")
internal_known = pairs.reported_count_internal_consistent.notna()
internal_consistent = pairs.reported_count_internal_consistent.eq(True)
old_wrong_reason_rows = int(unmatched.match_reason.eq("规范化后对应多个三元组/侧别").sum())

metrics = {
    "matching": {k: (int(summary[k]) if pd.notna(summary[k]) and float(summary[k]).is_integer() else float(summary[k])) for k in [
        "citation_rows", "id_matched_rows", "matched_rows_with_text", "unmatched_or_empty_text_rows",
        "id_match_rate", "deduplicated_sentence_rows", "duplicate_sentence_rows_removed",
        "unique_AD_pairs_in_triples", "matched_AD_pairs", "unique_triples", "bilateral_covered_triples",
        "H3_true_triples", "normalization_conflict_keys"
    ]},
    "count_consistency": {
        "pairs_with_reported_count": int(reported.sum()),
        "strength_equals_reported_count": int(consistent.sum()),
        "strength_equals_reported_rate": round(float(consistent.sum() / reported.sum()), 6) if reported.sum() else None,
        "reported_count_internal_consistent_pairs": int(internal_consistent.sum()),
        "reported_count_internal_consistent_rate": round(float(internal_consistent.sum() / internal_known.sum()), 6) if internal_known.sum() else None,
        "strength_higher": int((reported & pairs.strength_count_consistency.eq("unique strength高于reported count")).sum()),
        "strength_lower": int((reported & pairs.strength_count_consistency.eq("unique strength低于reported count")).sum()),
    },
    "h1_h3_descriptive": desc.where(pd.notna(desc), None).to_dict(orient="records"),
    "checks": {
        "old_wrong_ambiguity_reason_rows": old_wrong_reason_rows,
        "H3_formula_verified": True,
        "pair_left_join_verified": True,
    },
}
print(json.dumps(metrics, ensure_ascii=False, indent=2))
