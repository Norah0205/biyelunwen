from pathlib import Path
import math, sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BASE=Path(__file__).resolve().parent
INPUT=BASE/'配对主表_轻量.csv.gz'
MODE='function' if '功能' in BASE.name else 'position'
OUTPUT=BASE/('引用功能四组对照分析结果.xlsx' if MODE=='function' else '引用位置四组对照分析结果.xlsx')
SEED=20261020
GROUPS=['Q11','Q10','Q01','Q00']

def norm_id(s): return s.fillna('').astype(str).str.strip().str.replace(r'\.0$','',regex=True)
def norm_function(s):
    x=s.fillna('缺失').astype(str).str.strip().str.lower()
    mp={'background':'Background','discuss':'Discuss','discussion':'Discuss','basis':'Basis','support':'Support','differ':'Differ','difference':'Differ','nan':'缺失','':'缺失'}
    return x.map(mp).fillna(x.str.title())
def norm_section(s):
    x=s.fillna('缺失').astype(str).str.strip().str.lower()
    mp={'introduction':'Introduction','discussion':'Discussion','results':'Results','result':'Results','methods':'Methods','method':'Methods','nan':'缺失','':'缺失'}
    return x.map(mp).fillna(x.str.title())
def cramers_v(table):
    a=np.asarray(table,dtype=float)
    if a.size==0 or a.sum()==0:return np.nan
    chi2=chi2_contingency(a,correction=False)[0]; n=a.sum(); r,k=a.shape
    return math.sqrt(chi2/(n*max(1,min(k-1,r-1))))
def kappa(a,b,levels):
    tab=pd.crosstab(pd.Categorical(a,categories=levels),pd.Categorical(b,categories=levels),dropna=False).to_numpy()
    n=tab.sum()
    if n==0:return np.nan
    po=np.trace(tab)/n; pe=(tab.sum(0)*tab.sum(1)).sum()/(n*n)
    return (po-pe)/(1-pe) if pe<1 else np.nan
def gwet_ac1(a,b,levels):
    tab=pd.crosstab(pd.Categorical(a,categories=levels),pd.Categorical(b,categories=levels),dropna=False).to_numpy(dtype=float)
    n=tab.sum(); q=len(levels)
    if n==0 or q<=1:return np.nan
    po=np.trace(tab)/n; p=(tab.sum(0)+tab.sum(1))/(2*n); pe=(1-(p*p).sum())/(q-1)
    return (po-pe)/(1-pe) if pe<1 else np.nan
def pair_match(df,control,key_cols):
    q=df[df.quadrant.eq('Q11')].copy(); c=df[df.quadrant.eq(control)].copy()
    q=q.sample(frac=1,random_state=SEED); c=c.sample(frac=1,random_state=SEED+GROUPS.index(control))
    q['_rank']=q.groupby(key_cols,dropna=False).cumcount(); c['_rank']=c.groupby(key_cols,dropna=False).cumcount()
    keys=key_cols+['_rank']; m=q.merge(c,on=keys,suffixes=('_Q11','_'+control))
    return m

def summarize(df,left,right,levels,label):
    rows=[]
    for g in GROUPS:
        p=df[df.quadrant.eq(g)]; n=len(p); same=(p[left]==p[right])
        tab=pd.crosstab(pd.Categorical(p[left],categories=levels),pd.Categorical(p[right],categories=levels),dropna=False)
        rows.append({'样本口径':label,'组别':g,'配对数':n,'一致数':int(same.sum()),'一致率':same.mean() if n else np.nan,'CohenKappa':kappa(p[left],p[right],levels),'GwetAC1':gwet_ac1(p[left],p[right],levels),'CramersV':cramers_v(tab)})
    return pd.DataFrame(rows)

def main():
    if not INPUT.exists(): raise FileNotFoundError(INPUT)
    print('读取轻量配对主表……')
    df=pd.read_csv(INPUT,low_memory=False)
    df['文献A_WOSID']=norm_id(df['文献A_WOSID'])
    if MODE=='function':
        left,right='Classification1','Classification2'; df[left]=norm_function(df[left]); df[right]=norm_function(df[right]); levels=['Background','Discuss','Basis','Support','Differ','缺失']; title='引用功能'
        match_keys=['文献A_WOSID','Section1','Section2']
    else:
        left,right='Section1','Section2'; df[left]=norm_section(df[left]); df[right]=norm_section(df[right]); levels=['Introduction','Discussion','Results','Methods','缺失']; title='引用位置'
        match_keys=['文献A_WOSID']
    df['组合']=df[left]+' — '+df[right]
    df['是否一致']=df[left].eq(df[right])
    group_counts=df.groupby('quadrant').size().reindex(GROUPS,fill_value=0).rename('配对数').reset_index().rename(columns={'quadrant':'组别'})
    side=pd.concat([df[['quadrant',left]].rename(columns={left:'类别'}),df[['quadrant',right]].rename(columns={right:'类别'})],ignore_index=True)
    dist=side.groupby(['quadrant','类别']).size().rename('数量').reset_index(); dist['组内比例']=dist['数量']/dist.groupby('quadrant')['数量'].transform('sum')
    agree=summarize(df,left,right,levels,'全量')
    combo=df.groupby(['quadrant','组合']).size().rename('数量').reset_index(); combo['组内比例']=combo['数量']/combo.groupby('quadrant')['数量'].transform('sum'); combo=combo.sort_values(['quadrant','数量'],ascending=[True,False])
    comparisons=[]
    for c in GROUPS[1:]:
        a=df[df.quadrant.eq('Q11')]; b=df[df.quadrant.eq(c)]
        table=pd.crosstab(pd.concat([pd.Series(['Q11']*len(a)),pd.Series([c]*len(b))],ignore_index=True),pd.concat([a[left],b[left]],ignore_index=True))
        chi,p,dof,_=chi2_contingency(table,correction=False)
        comparisons.append({'口径':'全量单侧类别分布','比较':f'Q11 vs {c}','Q11样本量':len(a),'控制组样本量':len(b),'卡方值':chi,'自由度':dof,'p值':p,'CramersV':cramers_v(table)})
        tab2=pd.DataFrame({'Q11':[a['是否一致'].sum(),(~a['是否一致']).sum()],c:[b['是否一致'].sum(),(~b['是否一致']).sum()]},index=['一致','不一致']).T
        chi2,p2,dof2,_=chi2_contingency(tab2,correction=False)
        comparisons.append({'口径':'全量配对一致率','比较':f'Q11 vs {c}','Q11样本量':len(a),'控制组样本量':len(b),'卡方值':chi2,'自由度':dof2,'p值':p2,'CramersV':cramers_v(tab2)})
    matched_rows=[]; matched_audit=[]
    for c in GROUPS[1:]:
        m=pair_match(df,c,match_keys); matched_audit.append({'比较':f'Q11 vs {c}','匹配键':' + '.join(match_keys),'匹配对数':len(m),'Q11覆盖率':len(m)/(df.quadrant.eq('Q11').sum())})
        for role,suf in [('Q11','_Q11'),(c,'_'+c)]:
            vals1=m[left+suf]; vals2=m[right+suf]; same=vals1.eq(vals2)
            matched_rows.append({'比较':f'Q11 vs {c}','组别':role,'匹配配对数':len(m),'一致率':same.mean(),'CohenKappa':kappa(vals1,vals2,levels),'GwetAC1':gwet_ac1(vals1,vals2,levels)})
    matched=pd.DataFrame(matched_rows); matched_audit=pd.DataFrame(matched_audit)
    method=pd.DataFrame([['分析对象',title],['四组口径','Q11=高相似高耦合；Q10=高相似低耦合；Q01=低相似高耦合；Q00=双低'],['阈值','SBERT相似度≥0.60；耦合强度≥10'],['全量分析','保留2,085,004条配对记录'],['匹配分析',f'以Q11为基准，按{"、".join(match_keys)}及组内序号与各控制组1:1匹配'],['重要限制',f'匹配时不使用{title}，避免把结果变量提前控制一致'],['随机种子',SEED]],columns=['项目','说明'])
    with pd.ExcelWriter(OUTPUT,engine='openpyxl') as w:
        method.to_excel(w,'分析说明',index=False); group_counts.to_excel(w,'四组样本量',index=False); dist.to_excel(w,'单侧分布',index=False); agree.to_excel(w,'配对一致性',index=False); combo.head(200).to_excel(w,'功能组合Top' if MODE=='function' else '位置组合Top',index=False); pd.DataFrame(comparisons).to_excel(w,'全量组间比较',index=False); matched_audit.to_excel(w,'匹配覆盖率',index=False); matched.to_excel(w,'匹配后一致性',index=False); df[df.quadrant.eq('Q11')].head(10000).to_excel(w,'Q11审计样本',index=False)
    print('完成：',OUTPUT)
if __name__=='__main__': main()
