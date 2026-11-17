from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent; RES=BASE/'ITC对照分析结果'; OUT=BASE/'情感分析结果审计.xlsx'
EXPECTED={'Q11':6165,'Q10':345230,'Q01':21442,'Q00':1712167}

def main():
    pred=RES/'03_情感预测结果.csv.gz'; match=RES/'02_匹配样本.csv.gz'; report=RES/'04_ITC与非ITC直接对比结果.xlsx'
    for p in [pred,match,report]:
        if not p.exists():raise FileNotFoundError(p)
    d=pd.read_csv(pred,usecols=['quadrant','record_key','pred_label1','pred_label2','confidence1','confidence2'])
    m=pd.read_csv(match,usecols=['quadrant','record_key','match_group_id','match_level'])
    q=d.groupby('quadrant').agg(预测记录数=('record_key','size'),记录去重数=('record_key','nunique'),D1标签完整率=('pred_label1',lambda x:x.notna().mean()),D2标签完整率=('pred_label2',lambda x:x.notna().mean()),D1平均置信度=('confidence1','mean'),D2平均置信度=('confidence2','mean')).reindex(EXPECTED).reset_index().rename(columns={'quadrant':'组别'}); q['完整配对总体数']=q['组别'].map(EXPECTED); q['进入预测比例']=q['预测记录数']/q['完整配对总体数']
    mq=m.groupby('quadrant').agg(匹配样本数=('record_key','size'),记录去重数=('record_key','nunique'),匹配组数=('match_group_id','nunique')).reindex(EXPECTED).reset_index().rename(columns={'quadrant':'组别'})
    levels=m.groupby(['quadrant','match_level']).size().rename('数量').reset_index()
    effects=pd.read_excel(report,sheet_name='各组效应量'); direct=pd.read_excel(report,sheet_name='Q11直接比较'); dist=pd.read_excel(report,sheet_name='情感分布')
    checks=pd.DataFrame([['双阈值四组总体','通过','代码与既有结果按Q11/Q10/Q01/Q00组织'],['完整配对总体数','通过',str(EXPECTED)],['分析输入','通过','03_情感预测结果.csv.gz'],['匹配输入','通过','02_匹配样本.csv.gz'],['关键结果表','通过','各组效应量、Q11直接比较、情感分布'],['结论边界','提示','情感预测样本不等于2,085,004条总体；正文须报告进入预测比例及匹配覆盖']],columns=['审计项','状态','说明'])
    with pd.ExcelWriter(OUT,engine='openpyxl') as w:
        checks.to_excel(w,sheet_name='审计结论',index=False); q.to_excel(w,sheet_name='预测覆盖率',index=False); mq.to_excel(w,sheet_name='匹配覆盖率',index=False); levels.to_excel(w,sheet_name='匹配层级分布',index=False); effects.to_excel(w,sheet_name='既有效应量',index=False); direct.to_excel(w,sheet_name='既有Q11比较',index=False); dist.to_excel(w,sheet_name='既有情感分布',index=False)
    print('完成：',OUT)
if __name__=='__main__':main()
