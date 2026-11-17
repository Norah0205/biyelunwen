import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import cohen_kappa_score

import config
from core import save_excel, save_json


def safe_kappa(a, b, weights=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cohen_kappa_score(a, b, labels=config.LABELS, weights=weights)


def gwet_ac1(a, b):
    a, b = pd.Series(a), pd.Series(b)
    ok = a.isin(config.LABELS) & b.isin(config.LABELS)
    a, b = a[ok], b[ok]
    if not len(a): return np.nan
    po = float((a.to_numpy() == b.to_numpy()).mean())
    p = np.array([((a == lab).mean() + (b == lab).mean()) / 2 for lab in config.LABELS])
    pe = float(np.sum(p * (1 - p)) / (len(config.LABELS) - 1))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def _nonempty_table(table):
    """删除边际和为零的行列，避免稀疏小样本导致统计函数报错。"""
    return table.loc[table.sum(axis=1).gt(0), table.sum(axis=0).gt(0)]


def cramers_v(table):
    table = _nonempty_table(table)
    if table.empty or min(table.shape) < 2: return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1)))) if n else np.nan


def group_metrics(g):
    ok = g.pred_label1.isin(config.LABELS) & g.pred_label2.isin(config.LABELS)
    x = g[ok]
    if x.empty: return {"n": 0}
    table = pd.crosstab(x.pred_label1, x.pred_label2).reindex(index=config.LABELS, columns=config.LABELS, fill_value=0)
    chi_table = _nonempty_table(table)
    chi2_p = chi2_contingency(chi_table, correction=False)[1] if min(chi_table.shape, default=0) > 1 else np.nan
    rho = spearmanr(x.polarity_score1, x.polarity_score2, nan_policy="omit").statistic if len(x) > 1 else np.nan
    labels = pd.concat([x.pred_label1, x.pred_label2], ignore_index=True)
    result = {
        "n": len(x), "consistency_rate": x.pred_label1.eq(x.pred_label2).mean(),
        "kappa_unweighted": safe_kappa(x.pred_label1, x.pred_label2),
        "kappa_weighted_linear": safe_kappa(x.pred_label1, x.pred_label2, "linear"),
        "gwet_ac1": gwet_ac1(x.pred_label1, x.pred_label2), "cramers_v": cramers_v(table),
        "polarity_spearman": rho, "chi_square_p": chi2_p,
    }
    for lab in config.LABELS:
        result[f"proportion_{lab}"] = float((labels == lab).mean())
    return result


def _metric_table(t, kind):
    n = t.sum()
    if not n: return np.nan
    if kind == "consistency": return np.trace(t) / n
    if kind == "kappa":
        po, pe = np.trace(t) / n, np.sum(t.sum(1) * t.sum(0)) / n ** 2
        return (po - pe) / (1 - pe) if pe < 1 else np.nan
    i = config.LABELS.index(kind.replace("prop_", ""))
    return (t[i, :].sum() + t[:, i].sum()) / (2 * n)


def _common_match_subset(df, control):
    """只返回 Q11 与指定控制组都实际存在的共同匹配组。"""
    q11_groups = set(df.loc[df.quadrant.eq("Q11"), "match_group_id"].dropna())
    control_groups = set(df.loc[df.quadrant.eq(control), "match_group_id"].dropna())
    groups = sorted(q11_groups & control_groups)
    return df[df.quadrant.isin(["Q11", control]) & df.match_group_id.isin(groups)], groups


def confidence_intervals(df, control, reps=None):
    """在共同匹配组内按组重抽样，估计直接比较差值的 95% 置信区间。"""
    reps = config.INTERVAL_REPS if reps is None else reps
    x, groups = _common_match_subset(df, control)
    if not groups:
        return {}
    pos = {g: i for i, g in enumerate(groups)}
    cubes = {}
    for q in ("Q11", control):
        cube = np.zeros((len(groups), 3, 3), dtype=float)
        for r in x[x.quadrant.eq(q)].itertuples(index=False):
            if r.pred_label1 in config.LABELS and r.pred_label2 in config.LABELS:
                cube[pos[r.match_group_id], config.LABELS.index(r.pred_label1), config.LABELS.index(r.pred_label2)] += 1
        cubes[q] = cube
    kinds = ["consistency", "kappa", *[f"prop_{x}" for x in config.LABELS]]
    values = {k: [] for k in kinds}
    rng = np.random.default_rng(config.SEED)
    for _ in range(reps):
        idx = rng.integers(0, len(groups), len(groups))
        a, b = cubes["Q11"][idx].sum(0), cubes[control][idx].sum(0)
        for kind in kinds:
            values[kind].append(_metric_table(a, kind) - _metric_table(b, kind))
    return {k: (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))) for k, v in values.items()}


def direct_comparisons(df, metrics=None, reps=None):
    """每项直接比较严格限制在该控制组与 Q11 的共同 match_group_id。"""
    rows = []
    q11_total = df.loc[df.quadrant.eq("Q11"), "match_group_id"].nunique()
    for q in ("Q10", "Q01", "Q00"):
        paired, groups = _common_match_subset(df, q)
        if not groups:
            continue
        a = pd.Series(group_metrics(paired[paired.quadrant.eq("Q11")]))
        b = pd.Series(group_metrics(paired[paired.quadrant.eq(q)]))
        ci = confidence_intervals(paired, q, reps)
        row = {"comparison": f"Q11 vs {q}", "matched_pair_n": len(groups),
               "coverage_rate": len(groups) / q11_total if q11_total else np.nan,
               "q11_comparison_n": int(a.get("n", 0)), "control_comparison_n": int(b.get("n", 0)),
               "consistency_difference": a.consistency_rate - b.consistency_rate,
               "consistency_risk_ratio": a.consistency_rate / b.consistency_rate if b.consistency_rate else np.nan,
               "kappa_difference": a.kappa_unweighted - b.kappa_unweighted}
        for kind, outname in (("consistency", "consistency_diff"), ("kappa", "kappa_diff")):
            row[f"{outname}_ci_low"], row[f"{outname}_ci_high"] = ci.get(kind, (np.nan, np.nan))
        for lab in config.LABELS:
            av, bv = a[f"proportion_{lab}"], b[f"proportion_{lab}"]
            row[f"proportion_diff_{lab}"] = av - bv
            row[f"risk_ratio_{lab}"] = av / bv if bv else np.nan
            row[f"proportion_diff_{lab}_ci_low"], row[f"proportion_diff_{lab}_ci_high"] = ci.get(f"prop_{lab}", (np.nan, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(df):
    metrics = pd.DataFrame([{"quadrant": q, **group_metrics(g)} for q, g in df.groupby("quadrant")])
    order = [q for q in ("Q11", "Q10", "Q01", "Q00") if q in set(metrics.quadrant)]
    metrics["quadrant"] = pd.Categorical(metrics.quadrant, order, ordered=True)
    metrics = metrics.sort_values("quadrant").reset_index(drop=True)
    comparisons = direct_comparisons(df, metrics)
    dist = []
    for q, g in df.groupby("quadrant"):
        z = pd.concat([g.pred_label1, g.pred_label2]).value_counts(normalize=True)
        dist.extend({"quadrant": q, "label": lab, "proportion": float(z.get(lab, 0))} for lab in config.LABELS)
    dist = pd.DataFrame(dist)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in config.LABELS:
        z = dist[dist.label.eq(lab)].set_index("quadrant").reindex(order)
        ax.plot(order, z.proportion, marker="o", label=lab)
    ax.set(ylabel="比例", title="四组统一模型情感分布"); ax.legend(); fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "05_四组情感分布.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(metrics.quadrant.astype(str), metrics.consistency_rate)
    ax.set(ylim=(0, 1), ylabel="一致率", title="两引用句情感一致率"); fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "06_四组配对一致率.png", dpi=180); plt.close(fig)
    save_excel(config.OUTPUT_DIR / "04_ITC与非ITC直接对比结果.xlsx", {"各组效应量": metrics, "Q11直接比较": comparisons, "情感分布": dist})
    payload = {"各组效应量": metrics.astype(object).where(pd.notna(metrics), None).to_dict("records"), "Q11直接比较": comparisons.where(pd.notna(comparisons), None).to_dict("records"), "置信区间估计次数": config.INTERVAL_REPS, "说明": "四组均由同一训练模型预测；重点解释效应量及其95%置信区间。"}
    save_json(payload, config.OUTPUT_DIR / "04_ITC与非ITC直接对比结果.json")
    return payload
