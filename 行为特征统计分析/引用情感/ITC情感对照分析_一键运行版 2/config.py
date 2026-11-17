from pathlib import Path

# ===== 普通用户通常只需修改这一行 =====
DATA_DIR = Path("/Users/bytedance/Desktop/1013情感分析")

OUTPUT_DIR_NAME = "ITC对照分析结果"
OUTPUT_DIR = DATA_DIR / OUTPUT_DIR_NAME
FALLBACK_MODEL_DIR = Path("/Users/bytedance/Desktop/1013论文数据/itc_sentiment_pipeline/models/citation_sentiment_biomedbert")
VERSION = "v2"

SEED = 20261015
SBERT_THRESHOLD = 0.60
COUPLING_THRESHOLD = 10.0
CHUNK_SIZE = 100_000
CONTROLS_PER_ITC = 1
POOL_PER_STRATUM = 20
GLOBAL_POOL_PER_QUADRANT = 30_000
PRED_BATCH_SIZE = 16
MAX_LENGTH = 128
INTERVAL_REPS = 1_000
EXPECTED_TOTAL = 2_085_004
EXPECTED_COUNTS = {"Q11": 6_165, "Q10": 345_230, "Q01": 21_442, "Q00": 1_712_167}
LABELS = ["negative", "neutral", "positive"]
REQUIRED_COLUMNS = [
    "文献A_WOSID", "文献A题目", "文献D1_WOSID", "文献D1题目",
    "Section1", "Classification1", "引用句1", "文献D1参考文献",
    "文献D2_WOSID", "文献D2题目", "Section2", "Classification2", "引用句2",
    "文献D2参考文献", "耦合强度", "sbert相似度",
]
