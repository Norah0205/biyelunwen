"""合成回归测试：四组、安全分层实体匹配、缺失不可观测、统计与输出。"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import run_analysis as app

root = Path(__file__).parent / "_synthetic_test_v2"
if root.exists(): shutil.rmtree(root)
root.mkdir()

# 每组3个；Q00有额外样本。包含：ID可用、乱码ID靠题名、双通道冲突、无引用覆盖。
rows = []
groups = [(0.8,12),(0.8,5),(0.4,12),(0.4,5)]
for gi, (sim,coup) in enumerate(groups):
    for j in range(3 + (2 if gi == 3 else 0)):
        k = gi*10+j
        rows.append({"文献A_WOSID": "%06d"%(100000+k), "文献A_题目":"Target %d"%k,
                     "文献D1_WOSID":"%06d"%(200000+k), "文献D1_题目":"Citing B %d"%k,
                     "文献D2_WOSID":"%06d"%(300000+k), "文献D2_题目":"Citing C %d"%k,
                     "耦合强度":coup,"sbert相似度":sim,"A分组":"field%d"%(j%2)})
pairs = pd.DataFrame(rows)
# 令Q10第一条ID乱码，但题名可精确匹配。
pairs.loc[3,"文献D1_WOSID"] = "乱码-B"
pairs.to_csv(root/"配对关系-SBERT.csv",index=False,encoding="utf-8-sig")

cites=[]
for i,r in pairs.iterrows():
    # 最后一条Q00完全无覆盖，应为不可观测而不是0。
    if i == len(pairs)-1: continue
    for side in ("D1","D2"):
        count = 3 if (i%3==0) else 1
        for s in range(count):
            cites.append({"文献A_WOSID":r["文献A_WOSID"],"文献A_题目":r["文献A_题目"],
                          "文献D_WOSID":r["文献%s_WOSID"%side],"文献D_题目":r["文献%s_题目"%side],
                          "引用句":"Sentence %d"%s,"引用次数":count})
# WOSID冲突：ID指向第0个B pair，题名指向第1个B pair；必须保留题名匹配并标记审计冲突。
cites.append({"文献A_WOSID":pairs.loc[0,"文献A_WOSID"],"文献A_题目":pairs.loc[1,"文献A_题目"],
              "文献D_WOSID":pairs.loc[0,"文献D1_WOSID"],"文献D_题目":pairs.loc[1,"文献D1_题目"],"引用句":"conflict-retained","引用次数":1})
# 题名缺失，即便WOSID完整且正确也不可纳入。
cites.append({"文献A_WOSID":pairs.loc[0,"文献A_WOSID"],"文献A_题目":"",
              "文献D_WOSID":pairs.loc[0,"文献D1_WOSID"],"文献D_题目":pairs.loc[0,"文献D1_题目"],"引用句":"id-only-must-not-match","引用次数":1})
# 仅标点差异应进入第二层精确规范化，而不是模糊匹配。
cites.append({"文献A_WOSID":"错误ID-A","文献A_题目":"Target—0",
              "文献D_WOSID":"错误ID-D","文献D_题目":"Citing-B 0","引用句":"punctless-exact","引用次数":1})
# 模糊题名候选，不得进入正式结果。
cites.append({"文献A_WOSID":"坏ID","文献A_题目":"Target 0!","文献D_WOSID":"坏ID2","文献D_题目":"Citing B zero","引用句":"fuzzy","引用次数":1})
pd.DataFrame(cites).to_excel(root/"引用内容.xlsx",index=False)

out=root/"out"
r=app.run(root/"配对关系-SBERT.csv",root/"引用内容.xlsx",out)
t=r["triples"]
assert set(t.group)=={"Q11","Q10","Q01","Q00"}
assert len(t)==14
assert t.iloc[-1].bilateral_observable == False
assert np.isnan(t.iloc[-1].I_total) and np.isnan(t.iloc[-1].AB_strength)
# 乱码ID不影响：正式匹配始终由题名决定。
assert bool(t.iloc[3].AB_observable)
dedup = pd.read_csv(out/"引用句去重明细.csv")
assert "题名保守精确" in set(dedup["match_level"])
assert "题名去标点精确" in set(dedup["match_level"])
# WOSID与题名冲突保留题名结果并单列冲突；不能落入未匹配。
assert len(r["wosid_conflicts"]) == 1
assert "conflict-retained" in set(dedup.citation_text)
assert "conflict-retained" not in set(r["unmatched"].citation_text)
# 题名缺失绝不回退WOSID。
missing_title = r["unmatched"][r["unmatched"].citation_text.eq("id-only-must-not-match")]
assert len(missing_title) == 1 and missing_title.iloc[0].match_reason == "施引或被引题名缺失"
assert "id-only-must-not-match" not in set(dedup.citation_text)
assert len(r["primary"])==13
assert len(r["comparable"][r["comparable"].group.eq("Q11")])==3
assert all(len(r["comparable"][r["comparable"].group.eq(g)])<=3 for g in ["Q10","Q01","Q00"])
assert len(r["mw"])>0 and len(r["proportions"])>0
assert r["coverage"].loc[r["coverage"].group.eq("Q00"),"coverage_rate"].iloc[0] < 1
for name in ["ITC四组引用强度分析报告.xlsx","四组全量三元组.csv","可比样本_1比1.csv","ID质量检查.csv","WOSID与题名冲突明细.csv","字段识别诊断.csv","四组核心指标_可比样本.png","四组覆盖率.png"]:
    assert (out/name).exists(),name
print("合成测试通过：题名主键双层精确匹配、WOSID仅审计且冲突保留、题名缺失不回退、四组与统计输出均正常。")
