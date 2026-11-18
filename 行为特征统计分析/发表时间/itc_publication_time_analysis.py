"""
ITC发表时间分析（一键运行版）

本脚本直接读取同一文件夹中的两个Excel：
1. 隐形三角引用数据.xlsx：提供A、D1、D2配对关系
2. ITC文献发表时间与计算方案.xlsx：提供每篇文献的发表日期与回查链接

运行后生成：ITC发表时间分析结果.xlsx
首次运行请在PyCharm终端执行：pip install pandas openpyxl scipy
"""

from pathlib import Path
import re
import sys

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from openpyxl import load_workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError as exc:
    print("缺少依赖：", exc)
    print("请在PyCharm终端执行：pip install pandas openpyxl scipy")
    raise


# ==================== 已按你的文件路径填写 ====================
BASE_DIR = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path(r"/Users/bytedance/Desktop/1018发表时间")
ITC_FILE = BASE_DIR / "隐形三角引用数据.xlsx"
DATE_FILE = BASE_DIR / "ITC文献发表时间与计算方案.xlsx"
OUTPUT_FILE = BASE_DIR / "ITC发表时间分析结果.xlsx"
# =============================================================

ITC_SHEET = "Sheet1"
DATE_SHEET = "全部文献"
VALID_PRECISION = {"年月", "年月日"}


def text(value):
    return "" if pd.isna(value) else str(value).strip()


def normalize_wosid(value):
    value = text(value).replace("WOS:", "").replace("wos:", "")
    value = re.sub(r"\.0$", "", value)
    return re.sub(r"\s+", "", value)


def normalize_title(value):
    value = re.sub(r"<[^>]+>", "", text(value))
    value = value.replace("_", ":")
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def normalize_title_for_dedupe(value):
    # 与既有论文分析口径保持一致：直接对原始题名去除非字母数字字符。
    return re.sub(r"[^0-9a-z]+", "", text(value).lower())


def month_number(value):
    value = text(value)
    match = re.match(r"^(\d{4})[-/]?(\d{1,2})", value)
    if not match:
        return np.nan
    year, month = int(match.group(1)), int(match.group(2))
    if month < 1 or month > 12:
        return np.nan
    return year * 12 + month


def verify_columns(df, required, label):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(missing)}")


def build_lookup(date_df):
    records = date_df.to_dict("records")
    id_lookup, title_lookup = {}, {}
    for row in records:
        wid = normalize_wosid(row.get("原WOSID"))
        title_key = normalize_title(row.get("规范题名"))
        if wid and wid not in id_lookup:
            id_lookup[wid] = row
        if title_key and title_key not in title_lookup:
            title_lookup[title_key] = row
    return id_lookup, title_lookup


def match_document(wosid, title_value, id_lookup, title_lookup):
    """优先按WOSID匹配；WOSID无法匹配时，再按规范化题名匹配。"""
    wid = normalize_wosid(wosid)
    title_key = normalize_title(title_value)
    if wid and wid in id_lookup:
        return id_lookup[wid], "WOSID匹配"
    if title_key and title_key in title_lookup:
        return title_lookup[title_key], "题名匹配"
    return {}, "未匹配"


def describe(series, name):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "指标": name,
        "有效样本量": len(values),
        "均值": values.mean(),
        "标准差": values.std(),
        "最小值": values.min(),
        "P25": values.quantile(0.25),
        "中位数": values.median(),
        "P75": values.quantile(0.75),
        "最大值": values.max(),
        "单位": "月",
    }


def interval_distribution(series, bins, labels, metric):
    groups = pd.cut(series, bins=bins, labels=labels, right=True, include_lowest=True)
    result = groups.value_counts(sort=False).rename_axis("区间").reset_index(name="数量")
    result["比例"] = result["数量"] / result["数量"].sum()
    result.insert(0, "指标", metric)
    return result


def correlation_rows(valid):
    rows = []
    pairs = [
        ("平均引用时滞月数", "平均引用时滞", "sbert相似度", "SBERT相似度"),
        ("平均引用时滞月数", "平均引用时滞", "耦合强度", "耦合强度"),
        ("D—D时间差月数", "D—D时间差", "sbert相似度", "SBERT相似度"),
        ("D—D时间差月数", "D—D时间差", "耦合强度", "耦合强度"),
    ]
    for x_col, x_name, y_col, y_name in pairs:
        x = pd.to_numeric(valid[x_col], errors="coerce")
        y = pd.to_numeric(valid[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
            rho, p_value = spearmanr(x[mask], y[mask])
        else:
            rho, p_value = np.nan, np.nan
        rows.append({
            "变量1": x_name,
            "变量2": y_name,
            "有效样本量": int(mask.sum()),
            "Spearman ρ": rho,
            "p值": p_value,
        })
    return pd.DataFrame(rows)


def style_output(path):
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.sheet_view.showGridLines = False
        if ws.max_row >= 1 and ws.max_column >= 1:
            ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        widths = {}
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = Border(bottom=thin)
                if cell.value is not None:
                    widths[cell.column] = min(55, max(widths.get(cell.column, 10), len(str(cell.value)) * 1.05 + 2))
        for index, width in widths.items():
            ws.column_dimensions[get_column_letter(index)].width = width

    for sheet in ["样本筛选", "引用时滞区间分布", "D间隔区间分布", "核心比例指标"]:
        ws = wb[sheet]
        headers = {cell.value: cell.column for cell in ws[1]}
        for header in ["占原始样本比例", "比例"]:
            if header in headers:
                for row in range(2, ws.max_row + 1):
                    ws.cell(row, headers[header]).number_format = "0.00%"

    for sheet in ["时间指标描述统计", "相关性分析"]:
        ws = wb[sheet]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000"

    # 在两张分布表中增加原生柱状图。
    ws = wb["引用时滞区间分布"]
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "ITC引用时滞区间分布"
    chart.y_axis.title = "数量"
    chart.x_axis.title = "区间"
    data = Reference(ws, min_col=3, min_row=1, max_row=ws.max_row)
    cats = Reference(ws, min_col=2, min_row=2, max_row=ws.max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 8
    chart.width = 15
    ws.add_chart(chart, "F2")

    wb.save(path)


def main():
    print("=" * 70)
    print("ITC发表时间分析开始")
    print("=" * 70)

    for file_path in [ITC_FILE, DATE_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"未找到文件：{file_path}")

    print("[1/8] 读取两个Excel文件……")
    itc = pd.read_excel(ITC_FILE, sheet_name=ITC_SHEET)
    dates = pd.read_excel(DATE_FILE, sheet_name=DATE_SHEET)
    itc.insert(0, "原Excel行号", np.arange(2, len(itc) + 2))

    verify_columns(
        itc,
        ["文献A_WOSID", "文献A题目", "文献D1_WOSID", "文献D1题目", "文献D2_WOSID", "文献D2题目", "耦合强度", "sbert相似度"],
        "隐形三角引用数据",
    )
    verify_columns(
        dates,
        ["原WOSID", "规范题名", "分析采用年月", "日期精度", "匹配状态", "是否纳入月份分析", "回查链接"],
        "发表时间文件",
    )

    print(f"      ITC记录：{len(itc):,}条；文献日期：{len(dates):,}篇")
    print("[2/8] 将A、D1、D2与发表时间表匹配……")
    id_lookup, title_lookup = build_lookup(dates)

    for role, id_col, title_col in [
        ("A", "文献A_WOSID", "文献A题目"),
        ("D1", "文献D1_WOSID", "文献D1题目"),
        ("D2", "文献D2_WOSID", "文献D2题目"),
    ]:
        matched_rows, methods = [], []
        for wid, title_value in zip(itc[id_col], itc[title_col]):
            row, method = match_document(wid, title_value, id_lookup, title_lookup)
            matched_rows.append(row)
            methods.append(method)
        itc[f"{role}_匹配方式"] = methods
        for source, target in [
            ("规范题名", "规范题名"),
            ("DOI", "DOI"),
            ("原始发表日期", "原始发表日期"),
            ("日期精度", "日期精度"),
            ("分析采用年月", "分析采用年月"),
            ("数据源", "数据源"),
            ("匹配状态", "匹配状态"),
            ("是否纳入月份分析", "是否纳入月份分析"),
            ("回查链接", "回查链接"),
        ]:
            itc[f"{role}_{target}"] = [row.get(source, "") for row in matched_rows]
        itc[f"{role}_月份序号"] = itc[f"{role}_分析采用年月"].map(month_number)

    print("[3/8] 检查匹配率与日期完整性……")
    for role in ["A", "D1", "D2"]:
        itc[f"{role}_日期有效"] = (
            itc[f"{role}_匹配状态"].eq("高可信")
            & itc[f"{role}_日期精度"].isin(VALID_PRECISION)
            & itc[f"{role}_是否纳入月份分析"].eq("是")
            & itc[f"{role}_月份序号"].notna()
        )
    itc["三篇日期均有效"] = itc[["A_日期有效", "D1_日期有效", "D2_日期有效"]].all(axis=1)

    print("[4/8] 生成无序三元组键并去重……")
    def doc_key(wosid, title_value):
        # 去重以题名为主，避免原始WOSID格式差异或乱码改变三元组数量。
        return normalize_title_for_dedupe(title_value) or normalize_wosid(wosid)

    a_keys = [doc_key(w, t) for w, t in zip(itc["文献A_WOSID"], itc["文献A题目"])]
    d1_keys = [doc_key(w, t) for w, t in zip(itc["文献D1_WOSID"], itc["文献D1题目"])]
    d2_keys = [doc_key(w, t) for w, t in zip(itc["文献D2_WOSID"], itc["文献D2题目"])]
    itc["无序三元组键"] = [a + "||" + "||".join(sorted([d1, d2])) for a, d1, d2 in zip(a_keys, d1_keys, d2_keys)]
    itc["重复无序三元组"] = itc.duplicated("无序三元组键", keep="first")
    candidate = itc[itc["三篇日期均有效"] & ~itc["重复无序三元组"]].copy()

    print("[5/8] 计算四类时间指标……")
    candidate["D1距A月数"] = candidate["D1_月份序号"] - candidate["A_月份序号"]
    candidate["D2距A月数"] = candidate["D2_月份序号"] - candidate["A_月份序号"]
    candidate["最短引用时滞月数"] = candidate[["D1距A月数", "D2距A月数"]].min(axis=1)
    candidate["平均引用时滞月数"] = candidate[["D1距A月数", "D2距A月数"]].mean(axis=1)
    candidate["最长引用时滞月数"] = candidate[["D1距A月数", "D2距A月数"]].max(axis=1)
    candidate["D—D时间差月数"] = (candidate["D1_月份序号"] - candidate["D2_月份序号"]).abs()
    candidate["存在负时滞"] = (candidate["D1距A月数"] < 0) | (candidate["D2距A月数"] < 0)
    valid = candidate[~candidate["存在负时滞"]].copy()
    negative = candidate[candidate["存在负时滞"]].copy()

    print("[6/8] 计算描述统计、区间分布与相关性……")
    screening = pd.DataFrame({
        "筛选步骤": ["原始ITC记录", "A、D1、D2日期均有效", "剔除重复无序三元组后", "剔除负时滞后（主分析）"],
        "记录数": [len(itc), int(itc["三篇日期均有效"].sum()), len(candidate), len(valid)],
    })
    screening["占原始样本比例"] = screening["记录数"] / len(itc)

    descriptive = pd.DataFrame([
        describe(valid["最短引用时滞月数"], "最短引用时滞"),
        describe(valid["平均引用时滞月数"], "平均引用时滞"),
        describe(valid["最长引用时滞月数"], "最长引用时滞"),
        describe(valid["D—D时间差月数"], "两个施引文献发表时间差"),
    ])

    lag_bins = [-1e-9, 12, 36, 60, np.inf]
    lag_labels = ["0—12个月", "超过12—36个月", "超过36—60个月", "超过60个月"]
    lag_dist = pd.concat([
        interval_distribution(valid["最短引用时滞月数"], lag_bins, lag_labels, "最短引用时滞"),
        interval_distribution(valid["平均引用时滞月数"], lag_bins, lag_labels, "平均引用时滞"),
        interval_distribution(valid["最长引用时滞月数"], lag_bins, lag_labels, "最长引用时滞"),
    ], ignore_index=True)

    gap_dist = interval_distribution(
        valid["D—D时间差月数"],
        [-1e-9, 0, 12, 24, np.inf],
        ["同月", "1—12个月", "超过12—24个月", "超过24个月"],
        "D—D发表时间差",
    )

    cumulative = pd.DataFrame({
        "指标": ["平均引用时滞≤12个月", "平均引用时滞≤36个月", "平均引用时滞≤60个月", "D—D时间差≤12个月", "D—D时间差≤24个月"],
        "数量": [
            int((valid["平均引用时滞月数"] <= 12).sum()),
            int((valid["平均引用时滞月数"] <= 36).sum()),
            int((valid["平均引用时滞月数"] <= 60).sum()),
            int((valid["D—D时间差月数"] <= 12).sum()),
            int((valid["D—D时间差月数"] <= 24).sum()),
        ],
    })
    cumulative["比例"] = cumulative["数量"] / len(valid)
    correlations = correlation_rows(valid)

    match_audit = pd.DataFrame({
        "角色": ["A", "D1", "D2"],
        "WOSID匹配数": [(itc[f"{r}_匹配方式"] == "WOSID匹配").sum() for r in ["A", "D1", "D2"]],
        "题名匹配数": [(itc[f"{r}_匹配方式"] == "题名匹配").sum() for r in ["A", "D1", "D2"]],
        "未匹配数": [(itc[f"{r}_匹配方式"] == "未匹配").sum() for r in ["A", "D1", "D2"]],
        "日期有效数": [itc[f"{r}_日期有效"].sum() for r in ["A", "D1", "D2"]],
    })

    method = pd.DataFrame([
        ["月份序号", "M(X)=12×Year(X)+Month(X)"],
        ["D1距A月数", "M(D1)-M(A)"],
        ["D2距A月数", "M(D2)-M(A)"],
        ["最短引用时滞", "min(D1距A月数,D2距A月数)"],
        ["平均引用时滞", "(D1距A月数+D2距A月数)/2"],
        ["最长引用时滞", "max(D1距A月数,D2距A月数)"],
        ["D—D时间差", "|M(D1)-M(D2)|"],
        ["匹配顺序", "优先按WOSID匹配，无法匹配时按规范化题名匹配"],
        ["日期规则", "至少精确到年月；仅精确到年的文献不进入主分析"],
        ["去重规则", "同一A与无序D组合仅保留第一次出现的记录"],
        ["异常规则", "D1或D2早于A时标记为负时滞，不进入主分析"],
    ], columns=["项目", "口径"])

    print("[7/8] 输出可追溯Excel……")
    excluded = itc[~itc["三篇日期均有效"]].copy()
    duplicates = itc[itc["三篇日期均有效"] & itc["重复无序三元组"]].copy()
    trace_columns = [
        "原Excel行号", "文献A_WOSID", "文献A题目", "文献D1_WOSID", "文献D1题目", "文献D2_WOSID", "文献D2题目",
        "A_分析采用年月", "D1_分析采用年月", "D2_分析采用年月",
        "A_日期精度", "D1_日期精度", "D2_日期精度",
        "A_回查链接", "D1_回查链接", "D2_回查链接",
        "A_匹配方式", "D1_匹配方式", "D2_匹配方式",
        "D1距A月数", "D2距A月数", "最短引用时滞月数", "平均引用时滞月数", "最长引用时滞月数", "D—D时间差月数",
        "耦合强度", "sbert相似度", "无序三元组键",
    ]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        method.to_excel(writer, sheet_name="计算口径", index=False)
        match_audit.to_excel(writer, sheet_name="文献匹配审计", index=False)
        screening.to_excel(writer, sheet_name="样本筛选", index=False)
        descriptive.to_excel(writer, sheet_name="时间指标描述统计", index=False)
        lag_dist.to_excel(writer, sheet_name="引用时滞区间分布", index=False)
        gap_dist.to_excel(writer, sheet_name="D间隔区间分布", index=False)
        cumulative.to_excel(writer, sheet_name="核心比例指标", index=False)
        correlations.to_excel(writer, sheet_name="相关性分析", index=False)
        valid[trace_columns].to_excel(writer, sheet_name="主分析有效样本", index=False)
        negative.to_excel(writer, sheet_name="负时滞异常记录", index=False)
        excluded.to_excel(writer, sheet_name="日期无效或未匹配", index=False)
        duplicates.to_excel(writer, sheet_name="重复三元组记录", index=False)

    style_output(OUTPUT_FILE)

    print("[8/8] 完成")
    print("\n文献匹配审计：")
    print(match_audit.to_string(index=False))
    print("\n样本筛选：")
    print(screening.to_string(index=False))
    print("\n时间指标：")
    print(descriptive[["指标", "有效样本量", "均值", "中位数", "P25", "P75"]].round(3).to_string(index=False))
    print(f"\n输出文件：{OUTPUT_FILE}")
    print("主分析有效样本保留了原Excel行号、日期、回查链接和计算过程，可逐条溯源。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n运行失败：", error)
        print("请检查文件名、路径及Excel字段是否被修改。")
        sys.exit(1)
