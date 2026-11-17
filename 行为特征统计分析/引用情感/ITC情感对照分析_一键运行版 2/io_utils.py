"""自动识别文件并以受控内存方式读取。"""
from pathlib import Path
from typing import Iterator

import pandas as pd

import config
from core import log


def _supported(path):
    n = path.name.lower()
    return n.endswith((".csv", ".csv.gz", ".parquet", ".xlsx"))


def read_header(path):
    n = path.name.lower()
    if n.endswith((".csv", ".csv.gz")):
        return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    if n.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema.names
    if n.endswith(".xlsx"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        first = next(ws.iter_rows(values_only=True), ())
        wb.close()
        return [str(v).strip() if v is not None else "" for v in first]
    return []


def discover_files(data_dir=None):
    root = Path(data_dir or config.DATA_DIR)
    if not root.exists():
        raise FileNotFoundError(f"数据文件夹不存在：{root}\n请确认移动硬盘或桌面路径，并可在 config.py 顶部修改 DATA_DIR。")
    files = [p for p in sorted(root.iterdir()) if p.is_file() and _supported(p) and not p.name.startswith(("~$", ".")) and not p.name.startswith(config.OUTPUT_DIR_NAME)]
    preferred = [p for p in files if p.name == "隐形三角引用数据.xlsx"]
    named = [p for p in files if "itc" in p.name.lower()]
    itc = preferred[0] if preferred else (named[0] if named else None)
    if itc is None:
        raise FileNotFoundError("没有找到 ITC 文件。请把“隐形三角引用数据.xlsx”（或文件名含 ITC）放入数据目录。")
    raw, skipped = [], []
    for p in files:
        if p == itc:
            continue
        try:
            missing = sorted(set(config.REQUIRED_COLUMNS) - set(read_header(p)))
            if missing:
                skipped.append({"file": p.name, "reason": "缺少必需列", "missing": ", ".join(missing)})
                log(f"跳过 {p.name}：schema 不符，缺少 {len(missing)} 个必需列")
            else:
                raw.append(p)
        except Exception as exc:
            skipped.append({"file": p.name, "reason": f"读取表头失败：{exc}", "missing": ""})
            log(f"跳过 {p.name}：无法读取表头（{exc}）")
    if not raw:
        raise FileNotFoundError("未找到含全部 16 个必需列的完整原始数据文件。其他 Excel 已安全跳过，请检查列名。")
    log(f"识别 ITC 文件：{itc.name}；原始数据分片：{len(raw)} 个")
    return itc, raw, skipped


def iter_xlsx(path, chunk_size):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = [str(v).strip() if v is not None else "" for v in next(rows, ())]
    buf = []
    for row in rows:
        if not all(v is None for v in row):
            buf.append(row)
        if len(buf) >= chunk_size:
            yield pd.DataFrame(buf, columns=header); buf = []
    if buf:
        yield pd.DataFrame(buf, columns=header)
    wb.close()


def iter_file(path, chunk_size=None) -> Iterator[pd.DataFrame]:
    size = chunk_size or config.CHUNK_SIZE
    n = path.name.lower()
    if n.endswith((".csv", ".csv.gz")):
        yield from pd.read_csv(path, chunksize=size, low_memory=False, encoding="utf-8-sig")
    elif n.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=size):
            yield batch.to_pandas()
    elif n.endswith(".xlsx"):
        yield from iter_xlsx(path, size)


def iter_raw_files(paths, chunk_size=None):
    for path in paths:
        log(f"流式读取：{path.name}")
        for number, chunk in enumerate(iter_file(path, chunk_size), 1):
            yield path, number, chunk


def read_itc(path):
    parts = list(iter_file(path, config.CHUNK_SIZE))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def find_model(data_dir=None):
    root = Path(data_dir or config.DATA_DIR)
    candidates = sorted(p for p in root.rglob("citation_sentiment_biomedbert") if p.is_dir() and (p / "config.json").is_file())
    if candidates:
        return candidates[0]
    if config.FALLBACK_MODEL_DIR.is_dir() and (config.FALLBACK_MODEL_DIR / "config.json").is_file():
        return config.FALLBACK_MODEL_DIR
    raise FileNotFoundError(
        "没有找到训练好的情感模型。\n"
        "请把此前训练好的文件夹 citation_sentiment_biomedbert 完整复制到数据目录：\n"
        f"{root}\n"
        "该文件夹中必须含 config.json。程序不会重新训练，也不会下载大型模型。"
    )
