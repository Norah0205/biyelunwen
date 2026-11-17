import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from utils import prepare_sentence_inventory, save_excel_sheets, set_seed, validate_pair_data


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def predict_inventory(inventory):
    if not config.MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_DIR}. Run: python run_pipeline.py train"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_DIR).to(device)
    model.eval()

    logits_all = []
    texts = inventory["text"].tolist()
    for start in range(0, len(texts), config.PRED_BATCH_SIZE):
        batch = texts[start:start + config.PRED_BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=config.MAX_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits_all.append(model(**enc).logits.detach().cpu().numpy())
    probs = softmax(np.vstack(logits_all))

    # Always use config label order; verify the saved model mapping is compatible.
    result = inventory.copy()
    for i, label in enumerate(config.LABELS):
        result[f"p_{label}"] = probs[:, i]
    pred_idx = probs.argmax(axis=1)
    result["pred_label"] = [config.LABELS[i] for i in pred_idx]
    result["confidence"] = probs.max(axis=1)
    result["polarity_score"] = result["p_positive"] - result["p_negative"]
    result["review_reason"] = ""
    low = result["confidence"] < config.LOW_CONFIDENCE_THRESHOLD
    result.loc[low, "review_reason"] = "low_confidence"
    if config.REVIEW_ALL_NEGATIVE:
        neg = result["pred_label"].eq("negative")
        result.loc[neg & ~low, "review_reason"] = "negative_prediction"
        result.loc[neg & low, "review_reason"] = "low_confidence;negative_prediction"
    result["human_label"] = ""
    result["review_notes"] = ""
    return result


def build_review_sheet(pred):
    review = pred[pred["review_reason"].ne("")].copy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    eligible = pred[pred["review_reason"].eq("")]
    audits = []
    for label in ["neutral", "positive"]:
        pool = eligible[eligible["pred_label"].eq(label)]
        n = min(config.AUDIT_SAMPLE_PER_NONNEGATIVE_CLASS, len(pool))
        if n:
            audits.append(pool.loc[rng.choice(pool.index.to_numpy(), size=n, replace=False)].copy())
    if audits:
        audit = pd.concat(audits, ignore_index=True)
        audit["review_reason"] = "high_confidence_audit"
        review = pd.concat([review, audit], ignore_index=True)
    return review.sort_values(["review_reason", "confidence", "sent_id"]).reset_index(drop=True)


def build_validation_sample(pred):
    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    sampled = []
    mapping = []
    for label in config.LABELS:
        pool = pred[pred["pred_label"].eq(label)]
        n = min(config.VALIDATION_SAMPLE_PER_CLASS, len(pool))
        take = pool.loc[rng.choice(pool.index.to_numpy(), size=n, replace=False)].copy()
        sampled.append(take[["sent_id", "text"]])
        mapping.append(take[["sent_id", "pred_label", "confidence"]])
    blind = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    blind.insert(0, "validation_id", [f"V{i:04d}" for i in range(1, len(blind) + 1)])
    blind["human_label"] = ""
    blind["review_notes"] = ""
    key = blind[["validation_id", "sent_id"]].merge(pd.concat(mapping), on="sent_id", how="left")
    return blind, key


def run_prediction():
    set_seed()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair = pd.read_excel(config.INPUT_XLSX, sheet_name=config.SHEET_NAME)
    validate_pair_data(pair)
    pair.insert(0, "pair_row_id", np.arange(1, len(pair) + 1))
    long_df, inventory = prepare_sentence_inventory(pair)
    pred = predict_inventory(inventory)
    review = build_review_sheet(pred)
    validation_blind, validation_key = build_validation_sample(pred)

    pred.to_csv(config.OUTPUT_DIR / "01_sentence_predictions.csv", index=False, encoding="utf-8-sig")
    long_df.to_csv(config.OUTPUT_DIR / "01_sentence_pair_map.csv", index=False, encoding="utf-8-sig")
    validation_key.to_csv(config.OUTPUT_DIR / "03_validation_key.csv", index=False, encoding="utf-8-sig")
    save_excel_sheets(
        config.OUTPUT_DIR / "02_review_to_complete.xlsx",
        {"review": review, "instructions": pd.DataFrame({
            "instruction": [
                "Fill human_label with negative, neutral, positive, or uncertain.",
                "Review all low-confidence and negative predictions; do not delete rows.",
                "Use review_notes for ambiguous context or adjudication comments.",
            ]
        })},
    )
    save_excel_sheets(
        config.OUTPUT_DIR / "03_validation_to_complete.xlsx",
        {"validation": validation_blind, "instructions": pd.DataFrame({
            "instruction": [
                "This is a blinded validation sample. Do not open 03_validation_key.csv before labeling.",
                "Fill human_label with negative, neutral, positive, or uncertain.",
                "A second reviewer and adjudication are recommended for uncertain/disputed records.",
            ]
        })},
    )
    quality = pd.DataFrame({
        "metric": [
            "pair_records", "sentence_instances", "unique_sentences", "review_sentences",
            "low_confidence_unique", "predicted_negative_unique"
        ],
        "value": [
            len(pair), len(long_df), len(pred), len(review),
            int((pred.confidence < config.LOW_CONFIDENCE_THRESHOLD).sum()),
            int((pred.pred_label == "negative").sum()),
        ],
    })
    quality.to_csv(config.OUTPUT_DIR / "01_prediction_quality.csv", index=False, encoding="utf-8-sig")
    print(quality.to_string(index=False))
    print(f"Review file: {config.OUTPUT_DIR / '02_review_to_complete.xlsx'}")


if __name__ == "__main__":
    run_prediction()
