"""无需真实数据和模型的 v2 合成小数据测试。"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import config
from analysis import direct_comparisons, group_metrics
from audit import run_audit
from core import enrich
from io_utils import discover_files
from matching import build_candidate_pool, run_matching


def row(a, d1, d2, sim, coupling, section="Intro", cls="background", sentence1=None, sentence2=None):
    return {
        "文献A_WOSID": a, "文献A题目": f"A{a}", "文献D1_WOSID": d1, "文献D1题目": f"D{d1}",
        "Section1": section, "Classification1": cls, "引用句1": sentence1 or f"sentence {d1}", "文献D1参考文献": "r1",
        "文献D2_WOSID": d2, "文献D2题目": f"D{d2}", "Section2": section,
        "Classification2": cls, "引用句2": sentence2 or f"sentence {d2}", "文献D2参考文献": "r2",
        "耦合强度": coupling, "sbert相似度": sim,
    }


def test_files_quadrants_audit_and_matching():
    q11a, q11b = row(1, 101, 102, .80, 20), row(2, 201, 202, .70, 12)
    q01a = row(1, 105, 106, -.40, 15, sentence1="first context")
    q01b = row(1, 105, 106, -.40, 15, sentence1="different context")
    rows = [
        q11a, q11b, q11a.copy(),                         # exact duplicate target/raw
        row(1, 103, 104, .75, 3), row(2, 203, 204, .65, 7),
        q01a, q01b, q01a.copy(),                         # 同 pair_key 两条记录 + exact duplicate
        row(1, 107, 108, .20, 2), row(2, 207, 208, .50, 8),
    ]
    raw = pd.DataFrame(rows)
    enriched = enrich(raw)
    assert enriched.quadrant.value_counts().to_dict() == {"Q11": 3, "Q01": 3, "Q10": 2, "Q00": 2}
    q01 = enriched[enriched.quadrant.eq("Q01")]
    assert q01.pair_key.nunique() == 1 and q01.record_key.nunique() == 2
    # D1/D2 整侧交换后 record_key 仍稳定。
    swapped = q01a.copy()
    for c1, c2 in [("文献D1_WOSID", "文献D2_WOSID"), ("文献D1题目", "文献D2题目"),
                   ("Section1", "Section2"), ("Classification1", "Classification2"),
                   ("引用句1", "引用句2"), ("文献D1参考文献", "文献D2参考文献")]:
        swapped[c1], swapped[c2] = q01a[c2], q01a[c1]
    assert enrich(pd.DataFrame([q01a, swapped])).record_key.nunique() == 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); out = root / "ITC对照分析结果"
        raw.iloc[:5].to_csv(root / "part1.csv", index=False, encoding="utf-8-sig")
        raw.iloc[5:].to_excel(root / "part2.xlsx", index=False)
        pd.DataFrame({"无关列": [1]}).to_excel(root / "其他表.xlsx", index=False)
        pd.DataFrame([q11a, q11b, q11a.copy()]).to_excel(root / "隐形三角引用数据.xlsx", index=False)
        old = config.OUTPUT_DIR
        config.OUTPUT_DIR = out
        try:
            itc_path, raw_paths, skipped = discover_files(root)
            assert itc_path.name == "隐形三角引用数据.xlsx"
            assert len(raw_paths) == 2 and any(x["file"] == "其他表.xlsx" for x in skipped)
            expected = {"Q11": 3, "Q10": 2, "Q01": 3, "Q00": 2}
            ok = run_audit(itc_path, raw_paths, skipped, expected_total=10, expected_counts=expected)
            assert ok["passed"]
            assert len(ok["itc"]) == 2  # ITC target exact duplicate 已移除
            overview = ok["overview"].set_index("数据集")
            assert overview.loc["完整原始数据", "exact duplicate数"] == 2
            # 合法负相似度不计异常。
            numeric_check = next(x for x in ok["checks"] if x["核验项"].startswith("数值异常行"))
            assert numeric_check["实际值"] == 0 and numeric_check["结果"] == "通过"
            pool = build_candidate_pool(ok["itc"], raw_paths)
            pool_q01 = pool[pool.quadrant.eq("Q01")]
            assert len(pool_q01) == 2 and pool_q01.record_key.nunique() == 2
            assert pool_q01.pair_key.nunique() == 1
            sample = run_matching(ok["itc"], raw_paths)
            assert sample.quadrant.value_counts().to_dict() == {"Q11": 2, "Q10": 2, "Q01": 2, "Q00": 2}
            controls = sample[sample.quadrant.ne("Q11")]
            assert controls.groupby(["match_group_id", "quadrant"]).size().max() == 1
            assert controls.groupby("quadrant").record_key.nunique().eq(2).all()
            assert {"match_level", "distance", "record_key", "pair_key"}.issubset(sample.columns)
        finally:
            config.OUTPUT_DIR = old


def test_direct_statistics_common_match_groups():
    records = []
    patterns = {
        "Q11": [("positive", "positive"), ("negative", "negative"), ("neutral", "positive")],
        "Q10": [("positive", "neutral"), ("negative", "negative"), ("neutral", "negative")],
        # Q01 故意缺 M3；直接比较必须把 Q11 的 M3 同步排除。
        "Q01": [("positive", "positive"), ("negative", "neutral")],
        "Q00": [("positive", "negative"), ("negative", "positive"), ("neutral", "neutral")],
    }
    for q, pairs in patterns.items():
        for i, (a, b) in enumerate(pairs, 1):
            score = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
            records.append({"quadrant": q, "match_group_id": f"M{i}", "pred_label1": a, "pred_label2": b,
                            "polarity_score1": score[a], "polarity_score2": score[b]})
    df = pd.DataFrame(records)
    metrics = pd.DataFrame([{"quadrant": q, **group_metrics(g)} for q, g in df.groupby("quadrant")])
    result = direct_comparisons(df, metrics, reps=20)
    assert len(result) == 3
    assert {"consistency_difference", "kappa_difference", "risk_ratio_positive",
            "matched_pair_n", "coverage_rate"}.issubset(result.columns)
    q01 = result.set_index("comparison").loc["Q11 vs Q01"]
    assert q01.matched_pair_n == 2 and abs(q01.coverage_rate - 2 / 3) < 1e-12
    assert q01.q11_comparison_n == 2 and q01.control_comparison_n == 2
    assert abs(q01.consistency_difference - .5) < 1e-12
    assert result["consistency_diff_ci_low"].notna().all()


def main():
    test_files_quadrants_audit_and_matching()
    test_direct_statistics_common_match_groups()
    print("SMOKE TEST PASSED: record_key/exact duplicate/Q01同三元组多记录/负相似度/1:1匹配/共同match_group比较")


if __name__ == "__main__":
    main()
