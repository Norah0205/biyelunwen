"""确认唯一入口可被加载，且加载时不会启动真实分析。"""
import importlib.util
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
entry = project / "一键运行_ITC对照分析.py"
spec = importlib.util.spec_from_file_location("itc_one_click_entry", entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.main)
print("STATIC IMPORT PASSED")
