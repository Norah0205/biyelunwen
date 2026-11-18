"""
ITC学术影响力分析（一键运行版）

研究设计：
1. 不比较A与D，也不比较D1与D2谁更高。
2. 分别比较：
   - 曾经作为A参与ITC的文献 vs 从未作为A参与ITC的文献；
   - 曾经作为D参与ITC的文献 vs 从未作为D参与ITC的文献。
3. 正文主指标预先设定为：被引WOSCC、使用次数180天。
4. 被引ALLDB、使用次数总作为补充与稳健性指标，不根据显著性临时挑选指标。
5. 输出发表年份分层结果，用于检查累计指标的发表年限混杂。

准备文件（放在同一文件夹）：
- 原始文献A.xlsx
- 施引文献D.xlsx
- 隐形三角引用数据.xlsx

首次运行：pip install pandas openpyxl scipy
"""

from pathlib import Path
import re
import sys

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu, spearmanr
    from openpyxl import load_workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError as exc:
    print("缺少依赖：", exc)
    print("请在PyCharm终端执行：pip install pandas openpyxl scipy")
    raise


# ==================== 修改为你的文件夹 ====================
BASE_DIR = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path(r"/Users/bytedance/Desktop/1020学术影响力")
A_FILE = BASE_DIR / "原始文献A.xlsx"
D_FILE = BASE_DIR / "施引文献D.xlsx"
ITC_FILE = BASE_DIR / "隐形三角引用数据.xlsx"
OUTPUT_FILE = BASE_DIR / "ITC学术影响力分析结果.xlsx"
# ========================================================

A_SHEET = "原始文献A_有效保留"
D_SHEET = "施引文献D_有效保留"
ITC_SHEET = "Sheet1"

METRICS = ["使用次数180天", "使用次数总", "被引WOSCC", "被引ALLDB"]
PRIMARY_METRICS = ["被引WOSCC", "使用次数180天"]
SUPPLEMENT_METRICS = ["被引ALLDB", "使用次数总"]
MIN_YEAR_GROUP_N = 20


def text(value):
    return "" if pd.isna(value) else str(value).strip()


def normalize_wosid(value):
    value = text(value).replace("WOS:", "").replace("wos:", "")
    value = re.sub(r"\.0$", "", value)
    return re.sub(r"\s+", "", value)


def parse_publication_year(value):
    """兼容2022-03-01、MAR 23 2022、22-Feb、Feb-22等WoS常见格式。"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).year
    s = text(value)
    four_digit = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if four_digit:
        return int(four_digit.group(1))
    two_digit = re.search(r"(?:^|[-/ ])(\d{2})(?:[-/ ]|$)", s)
    if two_digit:
        yy = int(two_digit.group(1))
        if 0 <= yy <= 40:
            return 2000 + yy
        if 80 <= yy <= 99:
            return 1900 + yy
    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed):
            return int(parsed.year)
    except Exception:
        pass
    return np.nan


def require_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(missing)}")


def load_data():
    for path in [A_FILE, D_FILE, ITC_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"未找到文件：{path}")

    a = pd.read_excel(A_FILE, sheet_name=A_SHEET)
    d = pd.read_excel(D_FILE, sheet_name=D_SHEET)
    itc = pd.read_excel(ITC_FILE, sheet_name=ITC_SHEET)

    require_columns(a, ["WOSID", "题目", "发表时间"] + METRICS, "原始文献A")
    require_columns(d, ["WOSID", "题目", "发表时间"] + METRICS, "施引文献D")
    require_columns(
        itc,
        ["文献A_WOSID", "文献D1_WOSID", "文献D2_WOSID"],
        "隐形三角引用数据",
    )
    return a, d, itc


def prepare_documents(df, role, itc_ids):
    out = df.copy()
    out.insert(0, "原Excel行号", np.arange(2, len(out) + 2))
    out["WOSID标准化"] = out["WOSID"].map(normalize_wosid)
    out["发表年份"] = out["发表时间"].map(parse_publication_year)

    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")

    # 同一WOSID可能重复时保留第一条，同时输出重复审计。
    duplicate_mask = out["WOSID标准化"].duplicated(keep=False) | out["WOSID标准化"].eq("")
    duplicate_audit = out[duplicate_mask].copy()
    out = out[out["WOSID标准化"].ne("")].drop_duplicates("WOSID标准化", keep="first").copy()

    # 互斥分组：曾参与ITC vs 从未参与ITC。
    out["分组"] = np.where(out["WOSID标准化"].isin(itc_ids), "ITC", "非ITC")
    out["文献角色"] = role
    return out, duplicate_audit


def quality_table(df, role):
    rows = []
    for metric in METRICS:
        s = pd.to_numeric(df[metric], errors="coerce")
        rows.append({
            "文献角色": role,
            "指标": metric,
            "文献数": len(df),
            "有效数": int(s.notna().sum()),
            "缺失数": int(s.isna().sum()),
            "缺失比例": s.isna().mean(),
            "零值数": int(s.eq(0).sum()),
            "零值比例（有效值中）": s.eq(0).sum() / s.notna().sum() if s.notna().sum() else np.nan,
            "负值数": int((s < 0).sum()),
        })
    return pd.DataFrame(rows)


def logic_audit(df, role):
    checks = pd.DataFrame({
        "文献角色": role,
        "原Excel行号": df["原Excel行号"],
        "WOSID": df["WOSID"],
        "题目": df["题目"],
        "使用次数180天": df["使用次数180天"],
        "使用次数总": df["使用次数总"],
        "被引WOSCC": df["被引WOSCC"],
        "被引ALLDB": df["被引ALLDB"],
    })
    checks["180天使用次数大于总使用次数"] = checks["使用次数180天"] > checks["使用次数总"]
    checks["WOSCC被引大于ALLDB被引"] = checks["被引WOSCC"] > checks["被引ALLDB"]
    checks["存在负数"] = checks[METRICS].lt(0).any(axis=1)
    return checks[
        checks[["180天使用次数大于总使用次数", "WOSCC被引大于ALLDB被引", "存在负数"]].any(axis=1)
    ].copy()


def describe_groups(df, role):
    rows = []
    for metric in METRICS:
        for group in ["ITC", "非ITC"]:
            values = pd.to_numeric(df.loc[df["分组"].eq(group), metric], errors="coerce").dropna()
            rows.append({
                "文献角色": role,
                "指标": metric,
                "分组": group,
                "有效样本量": len(values),
                "零值比例": values.eq(0).mean() if len(values) else np.nan,
                "均值": values.mean(),
                "标准差": values.std(),
                "P25": values.quantile(0.25),
                "中位数": values.median(),
                "P75": values.quantile(0.75),
                "P90": values.quantile(0.90),
                "最大值": values.max(),
            })
    return pd.DataFrame(rows)


def delta_size(delta):
    value = abs(delta)
    if value < 0.147:
        return "可忽略"
    if value < 0.33:
        return "小"
    if value < 0.474:
        return "中等"
    return "大"


def compare_groups(df, role):
    rows = []
    for metric in METRICS:
        itc_values = pd.to_numeric(df.loc[df["分组"].eq("ITC"), metric], errors="coerce").dropna()
        non_values = pd.to_numeric(df.loc[df["分组"].eq("非ITC"), metric], errors="coerce").dropna()
        if len(itc_values) and len(non_values):
            u_stat, p_value = mannwhitneyu(itc_values, non_values, alternative="two-sided")
            cliff_delta = 2 * u_stat / (len(itc_values) * len(non_values)) - 1
        else:
            u_stat, p_value, cliff_delta = np.nan, np.nan, np.nan

        itc_median = itc_values.median() if len(itc_values) else np.nan
        non_median = non_values.median() if len(non_values) else np.nan
        direction = "ITC较高" if itc_median > non_median else "ITC较低" if itc_median < non_median else "中位数相同"
        rows.append({
            "文献角色": role,
            "指标": metric,
            "指标定位": "正文主指标" if metric in PRIMARY_METRICS else "补充指标",
            "ITC样本量": len(itc_values),
            "非ITC样本量": len(non_values),
            "ITC中位数": itc_median,
            "非ITC中位数": non_median,
            "中位数差（ITC-非ITC）": itc_median - non_median,
            "比较方向": direction,
            "Mann-Whitney U": u_stat,
            "p值": p_value,
            "Cliff's δ": cliff_delta,
            "效应量等级": delta_size(cliff_delta) if pd.notna(cliff_delta) else "无法计算",
            "统计显著": "是" if pd.notna(p_value) and p_value < 0.05 else "否",
        })
    return pd.DataFrame(rows)


def year_stratified(df, role):
    rows = []
    for metric in PRIMARY_METRICS:
        for year, part in df.dropna(subset=["发表年份"]).groupby("发表年份"):
            year = int(year)
            itc_values = pd.to_numeric(part.loc[part["分组"].eq("ITC"), metric], errors="coerce").dropna()
            non_values = pd.to_numeric(part.loc[part["分组"].eq("非ITC"), metric], errors="coerce").dropna()
            if len(itc_values) < MIN_YEAR_GROUP_N or len(non_values) < MIN_YEAR_GROUP_N:
                continue
            u_stat, p_value = mannwhitneyu(itc_values, non_values, alternative="two-sided")
            delta = 2 * u_stat / (len(itc_values) * len(non_values)) - 1
            rows.append({
                "文献角色": role,
                "指标": metric,
                "发表年份": year,
                "ITC样本量": len(itc_values),
                "非ITC样本量": len(non_values),
                "ITC中位数": itc_values.median(),
                "非ITC中位数": non_values.median(),
                "Cliff's δ": delta,
                "效应量等级": delta_size(delta),
                "p值": p_value,
            })
    return pd.DataFrame(rows)


def metric_correlations(df, role):
    rows = []
    for i, metric1 in enumerate(METRICS):
        for metric2 in METRICS[i + 1:]:
            x = pd.to_numeric(df[metric1], errors="coerce")
            y = pd.to_numeric(df[metric2], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                rho, p_value = spearmanr(x[mask], y[mask])
            else:
                rho, p_value = np.nan, np.nan
            rows.append({
                "文献角色": role,
                "指标1": metric1,
                "指标2": metric2,
                "有效样本量": int(mask.sum()),
                "Spearman ρ": rho,
                "p值": p_value,
            })
    return pd.DataFrame(rows)


def build_recommendation(comparisons, correlations):
    rows = []
    for metric in PRIMARY_METRICS + SUPPLEMENT_METRICS:
        metric_results = comparisons[comparisons["指标"].eq(metric)]
        strongest = metric_results.iloc[metric_results["Cliff's δ"].abs().argmax()] if len(metric_results) else None
        if metric in PRIMARY_METRICS:
            use = "正文保留"
            reason = "预先设定的代表性维度，不以显著性决定是否呈现"
        elif metric == "被引ALLDB":
            use = "补充表"
            reason = "与被引WOSCC口径重叠，用于检验结论是否稳健"
        else:
            use = "补充表"
            reason = "累计使用度受发表年限影响，作为近期使用指标的补充"
        rows.append({
            "指标": metric,
            "定位": use,
            "代表含义": "长期学术影响力" if metric == "被引WOSCC" else "近期关注度" if metric == "使用次数180天" else "扩展数据库被引" if metric == "被引ALLDB" else "累计使用度",
            "最大绝对Cliff's δ": abs(strongest["Cliff's δ"]) if strongest is not None else np.nan,
            "最大效应量等级": strongest["效应量等级"] if strongest is not None else "无法计算",
            "呈现理由": reason,
        })
    return pd.DataFrame(rows)


def style_workbook(path):
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.sheet_view.showGridLines = False
        if ws.max_row >= 1:
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
                    widths[cell.column] = min(50, max(widths.get(cell.column, 10), len(str(cell.value)) * 1.05 + 2))
        for index, width in widths.items():
            ws.column_dimensions[get_column_letter(index)].width = width

    for sheet in ["数据质量", "描述统计"]:
        ws = wb[sheet]
        headers = {c.value: c.column for c in ws[1]}
        for header in ["缺失比例", "零值比例（有效值中）", "零值比例"]:
            if header in headers:
                for row in range(2, ws.max_row + 1):
                    ws.cell(row, headers[header]).number_format = "0.00%"

    for sheet in ["组间比较", "年份分层", "指标相关性", "指标呈现建议"]:
        ws = wb[sheet]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000"

    # 为组间比较增加“ITC vs 非ITC中位数”图，只呈现预设的两个正文主指标。
    ws = wb["正文主指标图表数据"]
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "ITC与非ITC文献的主指标中位数"
    chart.y_axis.title = "中位数"
    chart.x_axis.title = "文献角色与指标"
    data = Reference(ws, min_col=3, max_col=4, min_row=1, max_row=ws.max_row)
    cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 8
    chart.width = 15
    ws.add_chart(chart, "F2")

    wb.save(path)


def main():
    print("=" * 72)
    print("ITC学术影响力分析开始")
    print("=" * 72)

    print("[1/8] 读取A、D与ITC文件……")
    a_raw, d_raw, itc = load_data()

    itc_a_ids = set(itc["文献A_WOSID"].map(normalize_wosid)) - {""}
    itc_d_ids = (set(itc["文献D1_WOSID"].map(normalize_wosid)) | set(itc["文献D2_WOSID"].map(normalize_wosid))) - {""}

    print("[2/8] 按文献角色建立互斥的ITC与非ITC组……")
    a, a_duplicates = prepare_documents(a_raw, "A", itc_a_ids)
    d, d_duplicates = prepare_documents(d_raw, "D", itc_d_ids)

    print("[3/8] 检查缺失值、零值和逻辑异常……")
    quality = pd.concat([quality_table(a, "A"), quality_table(d, "D")], ignore_index=True)
    logic_errors = pd.concat([logic_audit(a, "A"), logic_audit(d, "D")], ignore_index=True)
    duplicate_parts = []
    if not a_duplicates.empty:
        duplicate_parts.append(a_duplicates.assign(文献角色="A"))
    if not d_duplicates.empty:
        duplicate_parts.append(d_duplicates.assign(文献角色="D"))
    duplicate_audit = pd.concat(duplicate_parts, ignore_index=True) if duplicate_parts else pd.DataFrame()

    print("[4/8] 计算ITC与非ITC描述统计……")
    descriptive = pd.concat([describe_groups(a, "A"), describe_groups(d, "D")], ignore_index=True)

    print("[5/8] 进行Mann-Whitney U检验与Cliff's δ效应量分析……")
    comparisons = pd.concat([compare_groups(a, "A"), compare_groups(d, "D")], ignore_index=True)

    print("[6/8] 进行发表年份分层与指标冗余分析……")
    year_results = pd.concat([year_stratified(a, "A"), year_stratified(d, "D")], ignore_index=True)
    correlations = pd.concat([metric_correlations(a, "A"), metric_correlations(d, "D")], ignore_index=True)
    recommendation = build_recommendation(comparisons, correlations)

    group_counts = pd.concat([
        a.groupby("分组").size().rename("文献数").reset_index().assign(文献角色="A"),
        d.groupby("分组").size().rename("文献数").reset_index().assign(文献角色="D"),
    ], ignore_index=True)[["文献角色", "分组", "文献数"]]

    main_chart = comparisons[comparisons["指标"].isin(PRIMARY_METRICS)][
        ["文献角色", "指标", "ITC中位数", "非ITC中位数"]
    ].copy()
    main_chart["角色与指标"] = main_chart["文献角色"] + "-" + main_chart["指标"]
    main_chart = main_chart[["角色与指标", "指标", "ITC中位数", "非ITC中位数"]]

    method = pd.DataFrame([
        ["研究对象", "A与D分别分析，不进行A—D比较，也不比较D1与D2谁更高"],
        ["ITC组", "WOSID至少一次出现在ITC相应角色中的文献"],
        ["非ITC组", "存在于A或D文献全集、但从未出现在ITC相应角色中的文献"],
        ["去重单位", "同一角色内按标准化WOSID去重，每篇文献只进入一次"],
        ["正文主指标", "被引WOSCC（长期学术影响力）、使用次数180天（近期关注度）"],
        ["补充指标", "被引ALLDB、使用次数总"],
        ["描述口径", "数据右偏，以中位数、P25、P75和零值比例为主"],
        ["组间检验", "Mann-Whitney U双侧检验"],
        ["效应量", "Cliff's δ；绝对值<0.147为可忽略，<0.33为小，<0.474为中等，其余为大"],
        ["时间控制", f"对正文主指标按发表年份分层；每组每年至少{MIN_YEAR_GROUP_N}篇才报告"],
        ["呈现原则", "主指标预先设定；补充指标无论是否显著均保留在结果表，避免选择性报告"],
    ], columns=["项目", "分析口径"])

    print("[7/8] 写入可追溯Excel结果……")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        method.to_excel(writer, sheet_name="分析方案", index=False)
        group_counts.to_excel(writer, sheet_name="分组样本量", index=False)
        quality.to_excel(writer, sheet_name="数据质量", index=False)
        descriptive.to_excel(writer, sheet_name="描述统计", index=False)
        comparisons.to_excel(writer, sheet_name="组间比较", index=False)
        year_results.to_excel(writer, sheet_name="年份分层", index=False)
        correlations.to_excel(writer, sheet_name="指标相关性", index=False)
        recommendation.to_excel(writer, sheet_name="指标呈现建议", index=False)
        main_chart.to_excel(writer, sheet_name="正文主指标图表数据", index=False)
        logic_errors.to_excel(writer, sheet_name="逻辑异常记录", index=False)
        duplicate_audit.to_excel(writer, sheet_name="重复WOSID审计", index=False)
        a.to_excel(writer, sheet_name="A文献分组明细", index=False)
        d.to_excel(writer, sheet_name="D文献分组明细", index=False)

    style_workbook(OUTPUT_FILE)

    print("[8/8] 完成")
    print("\n分组样本量：")
    print(group_counts.to_string(index=False))
    print("\n组间比较：")
    print(comparisons[[
        "文献角色", "指标", "指标定位", "ITC中位数", "非ITC中位数",
        "比较方向", "p值", "Cliff's δ", "效应量等级"
    ]].round(4).to_string(index=False))
    print(f"\n逻辑异常记录：{len(logic_errors):,}条")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n运行失败：", error)
        print("请检查三个文件是否放在同一目录，文件名和工作表名是否与代码一致。")
        sys.exit(1)
