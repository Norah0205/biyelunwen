from collections import Counter
import sqlite3
import tempfile

import pandas as pd

import config
from core import enrich, log, save_excel
from io_utils import iter_raw_files, read_itc


def _insert_keys(conn, table, columns, rows):
    marks = ",".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({marks})",
        rows,
    )


def run_audit(itc_path, raw_paths, skipped, expected_total=None, expected_counts=None):
    """按行记录核验论文口径，并审计 record_key/pair_key 唯一性。"""
    expected_total = config.EXPECTED_TOTAL if expected_total is None else expected_total
    expected_counts = config.EXPECTED_COUNTS if expected_counts is None else expected_counts
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    itc_raw = read_itc(itc_path)
    missing = sorted(set(config.REQUIRED_COLUMNS) - set(itc_raw.columns))
    if missing:
        raise ValueError(f"ITC 文件缺少必需列：{missing}")
    itc_all = enrich(itc_raw[config.REQUIRED_COLUMNS])
    itc_original_n = len(itc_all)
    itc = itc_all.drop_duplicates("record_key").reset_index(drop=True)
    itc_exact_duplicates = itc_original_n - len(itc)
    unseen = set(itc.record_key)
    counts, file_rows = Counter(), Counter()
    numeric_bad = 0

    with tempfile.TemporaryDirectory(prefix="itc_audit_") as td:
        conn = sqlite3.connect(f"{td}/keys.sqlite")
        conn.executescript("""
            CREATE TABLE records (record_key TEXT PRIMARY KEY);
            CREATE TABLE pairs (pair_key TEXT PRIMARY KEY);
            CREATE TABLE quadrant_records (quadrant TEXT, record_key TEXT, PRIMARY KEY (quadrant, record_key));
            CREATE TABLE quadrant_pairs (quadrant TEXT, pair_key TEXT, PRIMARY KEY (quadrant, pair_key));
        """)
        for path, block, raw in iter_raw_files(raw_paths):
            x = enrich(raw[config.REQUIRED_COLUMNS])
            counts.update(x.quadrant.value_counts().to_dict())
            file_rows[path.name] += len(x)
            bad = (x["sbert相似度"].isna() | ~x["sbert相似度"].between(-1, 1)
                   | x["耦合强度"].isna() | x["耦合强度"].lt(0))
            numeric_bad += int(bad.sum())
            unseen.difference_update(x.record_key)
            _insert_keys(conn, "records", ["record_key"], ((v,) for v in x.record_key))
            _insert_keys(conn, "pairs", ["pair_key"], ((v,) for v in x.pair_key))
            _insert_keys(conn, "quadrant_records", ["quadrant", "record_key"], x[["quadrant", "record_key"]].itertuples(index=False, name=None))
            _insert_keys(conn, "quadrant_pairs", ["quadrant", "pair_key"], x[["quadrant", "pair_key"]].itertuples(index=False, name=None))
            conn.commit()
            log(f"口径核验 {path.name} 第 {block} 块，累计 {sum(file_rows.values()):,} 行")
        total = sum(file_rows.values())
        unique_records = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        unique_pairs = conn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        quadrant_unique_records = dict(conn.execute("SELECT quadrant, COUNT(*) FROM quadrant_records GROUP BY quadrant"))
        quadrant_unique_pairs = dict(conn.execute("SELECT quadrant, COUNT(*) FROM quadrant_pairs GROUP BY quadrant"))
        conn.close()

    itc_bad = itc[~itc.quadrant.eq("Q11")]
    checks = [
        {"核验项": "完整原始数据总数（行记录）", "实际值": total, "预期值": expected_total, "结果": "通过" if total == expected_total else "不通过"},
        {"核验项": "ITC原始总行数", "实际值": itc_original_n, "预期值": expected_counts["Q11"], "结果": "通过" if itc_original_n == expected_counts["Q11"] else "不通过"},
        {"核验项": "ITC全部满足阈值", "实际值": len(itc_bad), "预期值": 0, "结果": "通过" if itc_bad.empty else "不通过"},
        {"核验项": "ITC全部record_key包含于原始数据", "实际值": len(unseen), "预期值": 0, "结果": "通过" if not unseen else "不通过"},
        {"核验项": "数值异常行（相似度应在[-1,1]且耦合非负）", "实际值": numeric_bad, "预期值": 0, "结果": "通过" if numeric_bad == 0 else "不通过"},
    ]
    for q, expected in expected_counts.items():
        checks.append({"核验项": f"{q}数量（行记录）", "实际值": counts[q], "预期值": expected,
                       "结果": "通过" if counts[q] == expected else "不通过"})
    passed = all(r["结果"] == "通过" for r in checks)
    check_df = pd.DataFrame(checks)
    overview = pd.DataFrame([
        {"数据集": "完整原始数据", "原始总行数": total, "exact duplicate数": total - unique_records,
         "唯一record_key数": unique_records, "唯一pair_key数": unique_pairs},
        {"数据集": "ITC target", "原始总行数": itc_original_n, "exact duplicate数": itc_exact_duplicates,
         "唯一record_key数": itc.record_key.nunique(), "唯一pair_key数": itc.pair_key.nunique()},
    ])
    quadrant_summary = pd.DataFrame([
        {"象限": q, "行数": counts[q], "唯一record数": quadrant_unique_records.get(q, 0),
         "唯一三元组数": quadrant_unique_pairs.get(q, 0)}
        for q in ("Q11", "Q10", "Q01", "Q00")
    ])
    save_excel(config.OUTPUT_DIR / "01_数据口径核验.xlsx", {
        "核验结论": pd.DataFrame([{"总体结论": "通过" if passed else "不通过", "SBERT阈值": config.SBERT_THRESHOLD,
                                  "耦合强度阈值": config.COUPLING_THRESHOLD, "数值异常行": numeric_bad}]),
        "数据概况": overview,
        "四象限统计": quadrant_summary,
        "逐项核验": check_df,
        "原始分片": pd.DataFrame(file_rows.items(), columns=["文件", "有效行数"]),
        "跳过文件": pd.DataFrame(skipped, columns=["file", "reason", "missing"]),
        "ITC阈值问题": itc_bad,
        "ITC未包含明细": pd.DataFrame({"record_key": sorted(unseen)}),
    })
    for r in checks:
        log(f"{r['核验项']}：{r['结果']}（实际 {r['实际值']}，预期 {r['预期值']}）")
    return {"passed": passed, "itc": itc, "counts": dict(counts), "checks": checks,
            "overview": overview, "quadrant_summary": quadrant_summary}
