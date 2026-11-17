import numpy as np
import pandas as pd

import config
from core import log, normalized_sentence, stable_hash


def choose_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_prediction(sample, model_dir):
    """四组只使用同一个已训练模型，先规范化去重再回填。"""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sentences = pd.concat([
        sample[["pair_key", "引用句1"]].rename(columns={"引用句1": "text"}).assign(side=1),
        sample[["pair_key", "引用句2"]].rename(columns={"引用句2": "text"}).assign(side=2),
    ], ignore_index=True)
    sentences["text_norm"] = sentences.text.fillna("").map(normalized_sentence)
    inventory = sentences[["text_norm"]].drop_duplicates().reset_index(drop=True)
    inventory["sent_id"] = inventory.text_norm.map(lambda s: f"S{stable_hash(s, seed=1):016x}")
    valid = inventory.text_norm.ne("")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = choose_device(torch)
    model.to(device).eval()
    log(f"统一模型预测设备：{device}；规范化去重后 {len(inventory):,} 句")
    texts, blocks = inventory.loc[valid, "text_norm"].tolist(), []
    for start in tqdm(range(0, len(texts), config.PRED_BATCH_SIZE), desc="BiomedBERT 预测"):
        encoded = tokenizer(texts[start:start + config.PRED_BATCH_SIZE], padding=True, truncation=True, max_length=config.MAX_LENGTH, return_tensors="pt").to(device)
        with torch.inference_mode():
            blocks.append(torch.softmax(model(**encoded).logits, dim=1).cpu().numpy())
    probs = np.vstack(blocks) if blocks else np.empty((0, 3))
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    semantic = all(v in config.LABELS for v in id2label.values())
    for lab in config.LABELS:
        inventory[f"p_{lab}"] = np.nan
    for j, lab in enumerate(config.LABELS):
        model_j = next((k for k, v in id2label.items() if v == lab), j) if semantic else j
        inventory.loc[valid, f"p_{lab}"] = probs[:, model_j]
    pcols = [f"p_{lab}" for lab in config.LABELS]
    inventory["pred_label"] = inventory[pcols].idxmax(axis=1).str.removeprefix("p_")
    inventory["confidence"] = inventory[pcols].max(axis=1)
    inventory["polarity_score"] = inventory.p_positive - inventory.p_negative
    lookup = inventory.set_index("text_norm")
    result = sample.copy()
    for side in (1, 2):
        keys = result[f"引用句{side}"].fillna("").map(normalized_sentence)
        for col in ["sent_id", *pcols, "pred_label", "confidence", "polarity_score"]:
            result[f"{col}{side}"] = keys.map(lookup[col])
    result.to_csv(config.OUTPUT_DIR / "03_情感预测结果.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    log("预测完成：未混入既有人工标签或其他模型标签")
    return result
