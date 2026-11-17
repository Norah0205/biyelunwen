"""全流程编排；无命令行参数。"""
import traceback

import config
from analysis import run_analysis
from audit import run_audit
from core import log
from io_utils import discover_files, find_model
from matching import run_matching
from predict import run_prediction


def main():
    print("=" * 68)
    print(f"ITC 情感对照分析：一键运行（{config.VERSION}）")
    print(f"数据目录：{config.DATA_DIR}")
    print("=" * 68)
    try:
        itc_path, raw_paths, skipped = discover_files()
        audit = run_audit(itc_path, raw_paths, skipped)
        if not audit["passed"]:
            log("数据口径核验不通过。审计报告已完整生成，为避免错误分析，后续步骤已停止。")
            log(f"请先查看：{config.OUTPUT_DIR / '01_数据口径核验.xlsx'}")
            return 2
        model_dir = find_model()
        log(f"使用模型：{model_dir}")
        sample = run_matching(audit["itc"], raw_paths)
        predictions = run_prediction(sample, model_dir)
        run_analysis(predictions)
        print("\n分析全部完成。论文可直接使用的核心结果：")
        for name in ["01_数据口径核验.xlsx", "04_ITC与非ITC直接对比结果.xlsx", "04_ITC与非ITC直接对比结果.json", "05_四组情感分布.png", "06_四组配对一致率.png"]:
            print(f"  - {config.OUTPUT_DIR / name}")
        return 0
    except Exception as exc:
        print("\n运行未完成：", exc)
        print("\n给新手的排查建议：先检查数据目录、16个列名、requirements 和模型文件夹。")
        print("若仍无法解决，可把下面的错误详情发给技术同事：")
        traceback.print_exc()
        return 1
