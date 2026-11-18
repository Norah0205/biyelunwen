from pathlib import Path
import csv, gzip, json, sys, time

BASE = Path(__file__).resolve().parent
SRC = BASE / "配对关系-SBERT.csv"
OUT = BASE / "配对主表_轻量.csv.gz"
AUDIT = BASE / "配对主表_生成审计.json"
SIM_CUT = 0.60
COUPLING_CUT = 10.0
FIELDS = ["row_id","文献A_WOSID","文献D1_WOSID","文献D2_WOSID","Section1","Classification1","Section2","Classification2","耦合强度","sbert相似度","quadrant"]

def clean(x): return (x or "").strip()
def num(x):
    try: return float(x)
    except: return float("nan")
def quadrant(sim,coup):
    hi_s = sim >= SIM_CUT
    hi_c = coup >= COUPLING_CUT
    return "Q11" if hi_s and hi_c else "Q10" if hi_s else "Q01" if hi_c else "Q00"

def main():
    if not SRC.exists(): raise FileNotFoundError(SRC)
    csv.field_size_limit(sys.maxsize)
    counts={q:0 for q in ["Q11","Q10","Q01","Q00"]}; bad=0; total=0
    t=time.time()
    with SRC.open("r",encoding="utf-8-sig",newline="") as f, gzip.open(OUT,"wt",encoding="utf-8-sig",newline="",compresslevel=4) as g:
        r=csv.DictReader(f); missing=[c for c in FIELDS[1:-1] if c not in (r.fieldnames or [])]
        if missing: raise ValueError(f"缺少字段: {missing}")
        w=csv.DictWriter(g,fieldnames=FIELDS); w.writeheader()
        for i,row in enumerate(r,1):
            sim=num(row.get("sbert相似度")); coup=num(row.get("耦合强度"))
            if sim!=sim or coup!=coup: bad+=1; continue
            q=quadrant(sim,coup); counts[q]+=1; total+=1
            w.writerow({"row_id":i,"文献A_WOSID":clean(row.get("文献A_WOSID")),"文献D1_WOSID":clean(row.get("文献D1_WOSID")),"文献D2_WOSID":clean(row.get("文献D2_WOSID")),"Section1":clean(row.get("Section1")),"Classification1":clean(row.get("Classification1")),"Section2":clean(row.get("Section2")),"Classification2":clean(row.get("Classification2")),"耦合强度":coup,"sbert相似度":sim,"quadrant":q})
            if i%200000==0: print(f"已处理 {i:,} 行，用时 {(time.time()-t)/60:.1f} 分钟")
    audit={"源文件":str(SRC),"输出文件":str(OUT),"有效记录":total,"无效指标记录":bad,"阈值":{"sbert":SIM_CUT,"coupling":COUPLING_CUT},"四组记录数":counts,"耗时秒":round(time.time()-t,1)}
    AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
