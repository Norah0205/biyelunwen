import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import config
from utils import load_training_corpus, set_seed


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=config.MAX_LENGTH)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss


def metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def train_model():
    set_seed()
    df = load_training_corpus(config.TRAIN_CORPUS, config.TRAIN_TEXT_COL, config.TRAIN_LABEL_COL)
    label2id = {x: i for i, x in enumerate(config.LABELS)}
    df["label_id"] = df["label"].map(label2id)

    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["label_id"], random_state=config.RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=config.RANDOM_SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=len(config.LABELS),
        id2label={i: x for x, i in label2id.items()},
        label2id=label2id,
    )
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(config.LABELS)), y=train_df["label_id"]
    )

    training_kwargs = dict(
        output_dir=str(config.OUTPUT_DIR / "trainer_runs"),
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config.RANDOM_SEED,
        data_seed=config.RANDOM_SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )
    # Transformers changed the argument name across versions.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError("Installed transformers version has no supported evaluation strategy argument")
    args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        class_weights=torch.tensor(weights, dtype=torch.float),
        model=model,
        args=args,
        train_dataset=TextDataset(train_df.text, train_df.label_id, tokenizer),
        eval_dataset=TextDataset(val_df.text, val_df.label_id, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.train()
    pred = trainer.predict(TextDataset(test_df.text, test_df.label_id, tokenizer))
    y_pred = np.argmax(pred.predictions, axis=1)
    y_true = test_df.label_id.to_numpy()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MODEL_DIR))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus_records": len(df),
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "labels": config.LABELS,
        "metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "kappa": cohen_kappa_score(y_true, y_pred),
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=config.LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    with open(config.OUTPUT_DIR / "00_training_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train_model()
