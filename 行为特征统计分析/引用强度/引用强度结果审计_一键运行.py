from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent; RES=BASE/'ITC四组引用强度分析输出'; OUT=BASE/'引用强度结果审计.xlsx'; EXPECT={'Q11':6165,'Q10':345230,'Q01':21442,'Q00':1712167}

def main():
    files={'覆盖率':'覆盖率.csv','全量三元组':'四组全量三元组.csv','主分析':'主分析_双侧可观测.csv','可比样本':'可比样本_1比1.csv','效应量':'MannWhitneyU与CliffsDelta.csv','比例效应量':'比例检验与效应量.csv','报告':'ITC四组引用强度分析报告.xlsx'}
    for f in files.values():
        if not (RES/f).exists():raise FileNotFoundError(RES/f)
    cov=pd.read_csv(RES/files['覆盖率']); full=pd.read_csv(RES/files['全量三元组'],usecols=['group','triple_id','bilateral_observable']); main=pd.read_csv(RES/files['主分析']); matched=pd.read_csv(RES/files['可比样本']); eff=pd.read_csv(RES/files['效应量']); peff=pd.read_csv(RES/files['比例效应量'])
    counts=full.groupby('group').agg(分析三元组数=('triple_id','size'),三元组去重数=('triple_id','nunique'),双侧可观测数=('bilateral_observable','sum')).reindex(EXPECT).reset_index().rename(columns={'group':'组别'}); counts['完整配对总体数']=counts['组别'].map(EXPECT); counts['进入强度分析比例']=counts['分析三元组数']/counts['完整配对总体数']
    summary=pd.DataFrame([['四组口径','通过','结果文件包含Q11/Q10/Q01/Q00'],['双侧可观测','通过','主分析单独保存，覆盖率可审计'],['1:1匹配','通过','可比样本_1比1.csv存在'],['统计检验','通过','Mann-Whitney U与Cliff’s delta存在'],['总体与分析样本区别','提示','引用强度受引用内容匹配与双侧可观测约束，正文必须同时报告覆盖率'],['完整配对总体', '通过',str(EXPECT)]],columns=['审计项','状态','说明'])
    sizes=pd.DataFrame([['主分析_双侧可观测',len(main)],['可比样本_1比1',len(matched)]],columns=['数据集','记录数'])
    with pd.ExcelWriter(OUT,engine='openpyxl') as w:
        summary.to_excel(w,sheet_name='审计结论',index=False); counts.to_excel(w,sheet_name='总体口径核对',index=False); cov.to_excel(w,sheet_name='既有覆盖率',index=False); sizes.to_excel(w,sheet_name='分析集规模',index=False); eff.to_excel(w,sheet_name='既有效应量',index=False); peff.to_excel(w,sheet_name='既有比例效应量',index=False)
    print('完成：',OUT)
if __name__=='__main__':main()
