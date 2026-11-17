import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

import config
from utils import canonical_label, save_excel_sheets, validate_pair_data

LABEL_SCORE = {"negative": -1, "neutral": 0, "positive": 1}


def load_sheet(path, preferred):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    xls = pd.ExcelFile(path)
    name = preferred if preferred in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=name)


def apply_human_review(pred):
    review = load_sheet(config.REVIEW_COMPLETED_FILE, "review")
    required = {"sent_id", "human_label"}
    if not required.issubset(review.columns):
        raise ValueError(f"Review file requires columns: {required}")
    review["human_label"] = review["human_label"].map(canonical_label)
    valid = review[review["human_label"].isin(config.LABELS)].drop_duplicates("sent_id", keep="last")
    final = pred.merge(valid[["sent_id", "human_label"]], on="sent_id", how="left", suffixes=("", "_review"))
    # Prefer reviewed value imported from the completed workbook.
    reviewed_label = final["human_label_review"] if "human_label_review" in final else final["human_label"]
    final["final_label"] = reviewed_label.where(reviewed_label.isin(config.LABELS), final["pred_label"])
    final["label_source"] = np.where(reviewed_label.isin(config.LABELS), "human_review", "model")
    final["final_polarity_discrete"] = final["final_label"].map(LABEL_SCORE)
    return final


def evaluate_blinded_validation():
    completed = load_sheet(config.VALIDATION_COMPLETED_FILE, "validation")
    key = pd.read_csv(config.OUTPUT_DIR / "03_validation_key.csv", encoding="utf-8-sig")
    if not {"validation_id", "human_label"}.issubset(completed.columns):
        raise ValueError("Validation file requires validation_id and human_label")
    completed["human_label"] = completed["human_label"].map(canonical_label)
    val = completed.merge(key, on="validation_id", how="left", validate="one_to_one")
    val = val[val["human_label"].isin(config.LABELS)].copy()
    if len(val) < 30:
        raise ValueError("Too few completed validation records")
    report = classification_report(
        val["human_label"], val["pred_label"], labels=config.LABELS,
        output_dict=True, zero_division=0
    )
    metrics = {
        "n_labeled": len(val),
        "accuracy_on_stratified_sample": accuracy_score(val.human_label, val.pred_label),
        "macro_f1": f1_score(val.human_label, val.pred_label, average="macro"),
        "kappa_on_stratified_sample": cohen_kappa_score(val.human_label, val.pred_label),
        "classification_report": report,
        "confusion_matrix_order": config.LABELS,
        "confusion_matrix": confusion_matrix(val.human_label, val.pred_label, labels=config.LABELS).tolist(),
        "note": "Validation was stratified by predicted class; prioritize classwise metrics and macro-F1.",
    }
    return val, metrics


def cramers_v(table):
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    v = np.sqrt(chi2 / (n * min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else np.nan
    return chi2, p, dof, v


def bootstrap_cluster_ratio_ci(pair, numerator, denominator, reps=config.BOOTSTRAP_REPS):
    """Cluster bootstrap for a ratio without repeatedly concatenating full records."""
    tmp = pd.DataFrame({
        "triad_id": pair["triad_id"].to_numpy(),
        "num": np.asarray(numerator, dtype=float),
        "den": np.asarray(denominator, dtype=float),
    })
    grouped = tmp.groupby("triad_id", as_index=False)[["num", "den"]].sum()
    num = grouped["num"].to_numpy()
    den = grouped["den"].to_numpy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    values = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(grouped), size=len(grouped))
        values[i] = num[idx].sum() / den[idx].sum()
    lo, hi = np.quantile(values, [config.ALPHA / 2, 1 - config.ALPHA / 2])
    return lo, hi


def build_pair_data(sentence_final):
    pair = pd.read_excel(config.INPUT_XLSX, sheet_name=config.SHEET_NAME)
    validate_pair_data(pair)
    pair.insert(0, "pair_row_id", np.arange(1, len(pair) + 1))
    pair_map = pd.read_csv(config.OUTPUT_DIR / "01_sentence_pair_map.csv", encoding="utf-8-sig")
    mapped = pair_map.merge(
        sentence_final[["sent_id", "final_label", "final_polarity_discrete", "polarity_score", "confidence", "label_source"]],
        on="sent_id", how="left", validate="many_to_one"
    )
    for side in [1, 2]:
        cols = ["pair_row_id", "final_label", "final_polarity_discrete", "polarity_score", "confidence", "label_source"]
        s = mapped[mapped.side.eq(side)][cols].copy()
        s = s.rename(columns={c: f"{c}{side}" for c in cols if c != "pair_row_id"})
        pair = pair.merge(s, on="pair_row_id", how="left", validate="one_to_one")
    if pair[["final_label1", "final_label2"]].isna().any().any():
        raise ValueError("Some pair records failed sentiment linkage")
    pair["sentiment_consistent"] = (pair.final_label1 == pair.final_label2).astype(int)
    pair["mean_model_polarity"] = (pair.polarity_score1 + pair.polarity_score2) / 2
    pair["mean_discrete_polarity"] = (pair.final_polarity_discrete1 + pair.final_polarity_discrete2) / 2
    pair["min_confidence"] = pair[["confidence1", "confidence2"]].min(axis=1)
    pair["triad_id"] = pair.apply(
        lambda r: f"{r['文献A_WOSID']}|{'|'.join(sorted([str(r['文献D1_WOSID']), str(r['文献D2_WOSID'])]))}",
        axis=1,
    )
    return pair


def analyze(sentence_final, pair):
    labels = config.LABELS
    # Sentence-level distribution, unique and occurrence weighted.
    unique_dist = sentence_final.final_label.value_counts().reindex(labels, fill_value=0)
    instance_labels = pd.concat([pair.final_label1, pair.final_label2], ignore_index=True)
    instance_dist = instance_labels.value_counts().reindex(labels, fill_value=0)
    distribution = pd.DataFrame({
        "label": labels,
        "unique_sentence_n": unique_dist.values,
        "unique_sentence_pct": (unique_dist.values / unique_dist.sum() * 100),
        "sentence_instance_n": instance_dist.values,
        "sentence_instance_pct": (instance_dist.values / instance_dist.sum() * 100),
    })
    ci_rows = []
    for lab in labels:
        numerator = pair.final_label1.eq(lab).astype(int) + pair.final_label2.eq(lab).astype(int)
        denominator = np.full(len(pair), 2.0)
        lo, hi = bootstrap_cluster_ratio_ci(pair, numerator, denominator)
        ci_rows.append((lo * 100, hi * 100))
    distribution["cluster_bootstrap_ci_low"] = [x[0] for x in ci_rows]
    distribution["cluster_bootstrap_ci_high"] = [x[1] for x in ci_rows]

    combo = pd.crosstab(pair.final_label1, pair.final_label2).reindex(index=labels, columns=labels, fill_value=0)
    chi2, p, dof, v = cramers_v(combo)
    kappa = cohen_kappa_score(pair.final_label1, pair.final_label2)
    rho, rho_p = spearmanr(pair.polarity_score1, pair.polarity_score2)
    consistency = pair.sentiment_consistent.mean()
    ci_lo, ci_hi = bootstrap_cluster_ratio_ci(
        pair, pair.sentiment_consistent.to_numpy(), np.ones(len(pair))
    )

    pair_summary = pd.DataFrame({
        "metric": [
            "pair_records", "unique_document_triads", "sentiment_consistency_rate",
            "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high", "cohen_kappa",
            "label_association_chi2", "label_association_df", "label_association_p",
            "cramers_v", "model_score_spearman_rho", "model_score_spearman_p",
        ],
        "value": [
            len(pair), pair.triad_id.nunique(), consistency, ci_lo, ci_hi, kappa,
            chi2, dof, p, v, rho, rho_p,
        ],
    })

    # Section distributions for each role; do not pool them silently.
    section_tables = []
    context_assoc = []
    for side in [1, 2]:
        x = pd.crosstab(pair[f"Section{side}"], pair[f"final_label{side}"])
        x = x.reindex(columns=labels, fill_value=0)
        if x.shape[0] > 1:
            sx2, sp, sdf, sv = cramers_v(x)
            context_assoc.append({"side": side, "context_variable": "section", "chi2": sx2, "df": sdf, "p": sp, "cramers_v": sv})
        pct = x.div(x.sum(axis=1), axis=0) * 100
        x = x.reset_index().melt(id_vars=f"Section{side}", var_name="label", value_name="n")
        pct = pct.reset_index().melt(id_vars=f"Section{side}", var_name="label", value_name="pct")
        z = x.merge(pct, on=[f"Section{side}", "label"])
        z.insert(0, "side", side)
        z = z.rename(columns={f"Section{side}": "section"})
        section_tables.append(z)
    section_dist = pd.concat(section_tables, ignore_index=True)
    # Citation-function association, analyzed separately for the two sentence sides.
    for side in [1, 2]:
        ctab = pd.crosstab(pair[f"Classification{side}"].astype(str), pair[f"final_label{side}"])
        ctab = ctab.reindex(columns=labels, fill_value=0)
        if ctab.shape[0] > 1:
            cx2, cp, cdf, cv = cramers_v(ctab)
            context_assoc.append({"side": side, "context_variable": "citation_function", "chi2": cx2, "df": cdf, "p": cp, "cramers_v": cv})
    context_assoc = pd.DataFrame(context_assoc)
    context_assoc["note"] = "Exploratory chi-square; repeated document triads can reduce independence. Prioritize effect size."

    # Sensitivity by confidence cutoff. Human-reviewed labels remain included.
    sensitivity = []
    for cutoff in config.CONFIDENCE_SENSITIVITY_CUTOFFS:
        ok1 = (pair.confidence1 >= cutoff) | pair.label_source1.eq("human_review")
        ok2 = (pair.confidence2 >= cutoff) | pair.label_source2.eq("human_review")
        x = pair[ok1 & ok2]
        d = pd.concat([x.final_label1, x.final_label2]).value_counts(normalize=True).reindex(labels, fill_value=0)
        sensitivity.append({
            "confidence_cutoff": cutoff, "pair_n": len(x),
            "consistency_rate": x.sentiment_consistent.mean() if len(x) else np.nan,
            **{f"{lab}_pct": d[lab] * 100 for lab in labels},
        })
    sensitivity = pd.DataFrame(sensitivity)

    triad = (
        pair.groupby("triad_id", as_index=False)
        .agg(
            pair_records=("pair_row_id", "size"),
            mean_discrete_polarity=("mean_discrete_polarity", "mean"),
            mean_model_polarity=("mean_model_polarity", "mean"),
            sentiment_consistency_rate=("sentiment_consistent", "mean"),
        )
    )
    return distribution, combo, pair_summary, section_dist, context_assoc, sensitivity, triad


def finalize_and_analyze():
    pred = pd.read_csv(config.OUTPUT_DIR / "01_sentence_predictions.csv", encoding="utf-8-sig")
    sentence_final = apply_human_review(pred)
    validation_records, validation_metrics = evaluate_blinded_validation()
    pair = build_pair_data(sentence_final)
    distribution, combo, pair_summary, section_dist, context_assoc, sensitivity, triad = analyze(sentence_final, pair)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sentence_final.to_csv(config.OUTPUT_DIR / "04_sentence_final.csv", index=False, encoding="utf-8-sig")
    pair.to_excel(config.OUTPUT_DIR / "04_pair_level_final.xlsx", index=False)
    with open(config.OUTPUT_DIR / "04_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(validation_metrics, f, ensure_ascii=False, indent=2)
    save_excel_sheets(config.OUTPUT_DIR / "05_statistical_results.xlsx", {
        "sentiment_distribution": distribution,
        "label_combination": combo.reset_index(),
        "pair_summary": pair_summary,
        "section_distribution": section_dist,
        "context_association": context_assoc,
        "confidence_sensitivity": sensitivity,
        "triad_level": triad,
        "validation_records": validation_records,
    })
    print(pair_summary.to_string(index=False))
    print(distribution.to_string(index=False))


if __name__ == "__main__":
    finalize_and_analyze()
