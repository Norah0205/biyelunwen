from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE=Path(__file__).resolve().parent
PAIR=BASE/'配对主表_轻量.csv.gz'; AFILE=BASE/'原始文献A.xlsx'; DFILE=BASE/'施引文献D.xlsx'; OUT=BASE/'发表时间四组对照分析结果.xlsx'
GROUPS=['Q11','Q10','Q01','Q00']; SEED=20261020
METRICS=['最短引用时滞','平均引用时滞','最长引用时滞','D-D时间差']

def nid(s):
    x=s.fillna('').astype(str).str.strip().str.replace(r'\.0$','',regex=True).str.lstrip('0')
    return x.mask(x.eq(''),'0')
def month_index(s):
    dt=pd.to_datetime(s,errors='coerce')
    return dt.dt.year*12+dt.dt.month,dt.dt.year

def cliff_delta(x,y,block=2000):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)==0 or len(y)==0:return np.nan
    total=0
    for i in range(0,len(x),block):
        z=x[i:i+block,None]-y[None,:]
        total+=np.sign(z).sum()
    return total/(len(x)*len(y))
def level(d):
    a=abs(d)
    return '可忽略' if a<0.147 else '小' if a<0.33 else '中等' if a<0.474 else '大'
def compare(df,label):
    rows=[]; base=df[df.quadrant.eq('Q11')]
    for c in GROUPS[1:]:
        ctrl=df[df.quadrant.eq(c)]
        for m in METRICS:
            x=base[m].dropna().to_numpy(); y=ctrl[m].dropna().to_numpy()
            u,p=mannwhitneyu(x,y,alternative='two-sided') if len(x) and len(y) else (np.nan,np.nan)
            d=cliff_delta(x,y)
            rows.append({'口径':label,'比较':f'Q11 vs {c}','指标':m,'Q11样本量':len(x),'控制组样本量':len(y),'Q11中位数':np.median(x) if len(x) else np.nan,'控制组中位数':np.median(y) if len(y) else np.nan,'MannWhitneyU':u,'p值':p,"Cliff's delta":d,'效应量等级':level(d) if d==d else ''})
    return pd.DataFrame(rows)
def matched(df,control):
    q=df[df.quadrant.eq('Q11')].sample(frac=1,random_state=SEED).copy(); c=df[df.quadrant.eq(control)].sample(frac=1,random_state=SEED+GROUPS.index(control)).copy()
    q['_rank']=q.groupby('文献A_WOSID').cumcount(); c['_rank']=c.groupby('文献A_WOSID').cumcount()
    m=q.merge(c,on=['文献A_WOSID','_rank'],suffixes=('_Q11','_'+control))
    rows=[]
    for metric in METRICS:
        x=m[metric+'_Q11'].dropna(); y=m[metric+'_'+control].dropna(); both=m[[metric+'_Q11',metric+'_'+control]].dropna(); x=both.iloc[:,0].to_numpy(); y=both.iloc[:,1].to_numpy()
        u,p=mannwhitneyu(x,y,alternative='two-sided') if len(x) else (np.nan,np.nan); d=cliff_delta(x,y)
        rows.append({'口径':'同A 1:1匹配','比较':f'Q11 vs {control}','指标':metric,'匹配对数':len(x),'Q11覆盖率':len(m)/max(1,(df.quadrant.eq('Q11').sum())),'Q11中位数':np.median(x) if len(x) else np.nan,'控制组中位数':np.median(y) if len(y) else np.nan,'p值':p,"Cliff's delta":d,'效应量等级':level(d) if d==d else ''})
    return rows

def main():
    for p in [PAIR,AFILE,DFILE]:
        if not p.exists():raise FileNotFoundError(p)
    print('读取文献日期……')
    a=pd.read_excel(AFILE,sheet_name=0,usecols=['WOSID','发表时间']); d=pd.read_excel(DFILE,sheet_name=0,usecols=['WOSID','发表时间'])
    a['WOSID']=nid(a['WOSID']); d['WOSID']=nid(d['WOSID']); a=a.drop_duplicates('WOSID'); d=d.drop_duplicates('WOSID')
    a['A月序'],a['A年份']=month_index(a['发表时间']); d['D月序'],d['D年份']=month_index(d['发表时间'])
    amap=a.set_index('WOSID')[['A月序','A年份']]; dmap=d.set_index('WOSID')[['D月序','D年份']]
    print('读取配对主表并匹配日期……')
    df=pd.read_csv(PAIR,usecols=['文献A_WOSID','文献D1_WOSID','文献D2_WOSID','quadrant'],dtype=str)
    for c in ['文献A_WOSID','文献D1_WOSID','文献D2_WOSID']:df[c]=nid(df[c])
    # D1/D2无序去重
    df['D低']=df[['文献D1_WOSID','文献D2_WOSID']].min(axis=1); df['D高']=df[['文献D1_WOSID','文献D2_WOSID']].max(axis=1)
    before=len(df); df=df.drop_duplicates(['文献A_WOSID','D低','D高','quadrant']).copy(); dup=before-len(df)
    df=df.join(amap,on='文献A_WOSID').join(dmap.rename(columns={'D月序':'D1月序','D年份':'D1年份'}),on='文献D1_WOSID').join(dmap.rename(columns={'D月序':'D2月序','D年份':'D2年份'}),on='文献D2_WOSID')
    df['L1']=df['D1月序']-df['A月序']; df['L2']=df['D2月序']-df['A月序']; df['最短引用时滞']=df[['L1','L2']].min(axis=1); df['平均引用时滞']=df[['L1','L2']].mean(axis=1); df['最长引用时滞']=df[['L1','L2']].max(axis=1); df['D-D时间差']=(df['D1月序']-df['D2月序']).abs()
    df['日期完整']=df[['A月序','D1月序','D2月序']].notna().all(axis=1); df['存在负时滞']=(df[['L1','L2']].min(axis=1)<0)
    valid=df[df['日期完整']].copy()
    coverage=df.groupby('quadrant').agg(去重三元组=('quadrant','size'),日期完整数=('日期完整','sum'),负时滞数=('存在负时滞','sum')).reindex(GROUPS).reset_index().rename(columns={'quadrant':'组别'}); coverage['日期完整率']=coverage['日期完整数']/coverage['去重三元组']
    desc=valid.groupby('quadrant')[METRICS].describe(percentiles=[.25,.5,.75,.9]).stack(0).reset_index().rename(columns={'quadrant':'组别','level_1':'指标'})
    full=compare(valid,'全量无序三元组')
    mr=[]
    for c in GROUPS[1:]:mr.extend(matched(valid,c))
    matched_result=pd.DataFrame(mr)
    intervals=[]
    for g,p in valid.groupby('quadrant'):
        for m in METRICS:
            x=p[m]
            for label,mask in [('≤12个月',x<=12),('≤24个月',x<=24),('≤36个月',x<=36),('>60个月',x>60)]: intervals.append({'组别':g,'指标':m,'区间':label,'数量':int(mask.sum()),'比例':mask.mean()})
    method=pd.DataFrame([['分析单位','A固定、D1/D2无序三元组'],['四组口径','Q11/Q10/Q01/Q00来自完整2,085,004条配对数据'],['日期来源','原始文献A.xlsx与施引文献D.xlsx中的发表时间'],['全量比较','四组无序三元组'],['匹配比较','以Q11为基准，按相同A与三个控制组1:1匹配'],['统计方法','Mann-Whitney U + Cliff’s delta'],['重复删除数',dup],['限制','发表日期不能直接判定D1/D2的B/C角色']],columns=['项目','说明'])
    with pd.ExcelWriter(OUT,engine='openpyxl') as w:
        method.to_excel(w,sheet_name='分析说明',index=False); coverage.to_excel(w,sheet_name='日期覆盖率',index=False); desc.to_excel(w,sheet_name='描述统计',index=False); full.to_excel(w,sheet_name='全量组间比较',index=False); matched_result.to_excel(w,sheet_name='同A匹配比较',index=False); pd.DataFrame(intervals).to_excel(w,sheet_name='区间比例',index=False); valid[valid.quadrant.eq('Q11')].head(10000).to_excel(w,sheet_name='Q11审计样本',index=False); df[~df['日期完整']].head(50000).to_excel(w,sheet_name='日期缺失审计',index=False)
    print('完成：',OUT)
if __name__=='__main__':main()
