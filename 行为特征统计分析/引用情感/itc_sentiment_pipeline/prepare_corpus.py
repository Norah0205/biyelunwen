import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

import config
from utils import canonical_label, normalize_text

OFFICIAL_URLS = [
    "https://awaisathar.com/citation-sentiment-corpus/citation_sentiment_corpus.zip",
    "https://cl.awaisathar.com/citation-sentiment-corpus/citation_sentiment_corpus.zip",
]


def download_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def parse_corpus_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        files = [name for name in archive.namelist() if not name.endswith("/")]
        if not files:
            raise ValueError("Downloaded archive contains no data file")
        raw = archive.read(files[0])
    # Official corpus is tab-separated with an 18-line metadata header.
    df = pd.read_csv(
        io.BytesIO(raw), sep="\t", header=None, skiprows=18,
        names=["source_paper_id", "target_paper_id", "sentiment", "text"],
        encoding="utf-8", on_bad_lines="skip",
    )
    return df


def prepare_corpus():
    config.TRAIN_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    content = None
    used_url = None
    for url in OFFICIAL_URLS:
        try:
            content = download_bytes(url)
            if len(content) < 1000:
                raise ValueError("Downloaded file is unexpectedly small")
            used_url = url
            break
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    if content is None:
        raise RuntimeError(
            "Unable to download the official Athar corpus. Check network access and retry.\n"
            + "\n".join(errors)
        )

    raw = parse_corpus_zip(content)
    raw["label"] = raw["sentiment"].map({"n": "negative", "o": "neutral", "p": "positive"})
    raw["label"] = raw["label"].fillna(raw["sentiment"].map(canonical_label))
    clean = raw[["text", "label"]].copy()
    clean["text"] = clean["text"].astype(str).str.strip()
    clean = clean[clean["label"].isin(config.LABELS)]
    clean = clean[clean["text"].str.len().between(20, 600)]
    clean["norm_key"] = clean["text"].map(normalize_text)
    clean = clean.drop_duplicates("norm_key").drop(columns="norm_key").reset_index(drop=True)
    clean.to_csv(config.TRAIN_CORPUS, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame({
        "metric": ["download_url", "raw_records", "clean_records", "negative", "neutral", "positive"],
        "value": [
            used_url, len(raw), len(clean),
            int((clean.label == "negative").sum()),
            int((clean.label == "neutral").sum()),
            int((clean.label == "positive").sum()),
        ],
    })
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.OUTPUT_DIR / "00_corpus_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Training corpus saved to: {config.TRAIN_CORPUS}")


if __name__ == "__main__":
    prepare_corpus()
