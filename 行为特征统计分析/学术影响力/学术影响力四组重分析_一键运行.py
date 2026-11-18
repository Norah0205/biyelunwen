from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE=Path(__file__).resolve().parent
PAIR=BASE/'配对主表_轻量.csv.gz'; AFILE=BASE/'原始文献A.xlsx'; DFILE=BASE/'施引文献D.xlsx'; OUT=BASE/'学术影响力四组重分析结果.xlsx'
GROUPS=['Q11','Q10','Q01','Q00']; RAW=['使用次数180天','使用次数总','被引WOSCC','被引ALLDB']; SEED=20261020

def nid(s):
    x=s.fillna('').astype(str).str.strip().str.replace(r'\.0$','',regex=True).str.lstrip('0')
    return x.mask(x.eq(''),'0')
def prep(path):
    x=pd.read_excel(path,sheet_name=0,usecols=['WOSID']+RAW); x['WOSID']=nid(x.WOSID)
    for c in RAW:x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.drop_duplicates('WOSID').set_index('WOSID')
def effect(x,y):
    x=pd.Series(x).dropna().to_numpy(); y=pd.Series(y).dropna().to_numpy()
    if not len(x) or not len(y):return len(x),len(y),np.nan,np.nan,np.nan
    u,p=mannwhitneyu(x,y,alternative='two-sided'); delta=2*u/(len(x)*len(y))-1
    return len(x),len(y),u,p,delta
def grade(d):
    a=abs(d)
    return '可忽略' if a<.147 else '小' if a<.33 else '中等' if a<.474 else '大'
def compare(base,ctrl,label,metrics):
    rows=[]
    for m in metrics:
        n1,n0,u,p,d=effect(base[m],ctrl[m]); rows.append({'口径':label,'指标':m,'实验组样本量':n1,'对照组样本量':n0,'实验组均值':base[m].mean(),'实验组中位数':base[m].median(),'对照组均值':ctrl[m].mean(),'对照组中位数':ctrl[m].median(),'MannWhitneyU':u,'p值':p,"Cliff's delta":d,'效应量等级':grade(d) if d==d else ''})
    return rows
def doc_role(pairs,meta,idcols,role):
    q11=set(pd.concat([pairs.loc[pairs.quadrant.eq('Q11'),c] for c in idcols]).dropna())
    non=set(pd.concat([pairs.loc[~pairs.quadrant.eq('Q11'),c] for c in idcols]).dropna())
    allids=q11|non; rows=[]
    for i in allids:
        rows.append({'WOSID':i,'角色':role,'是否参与Q11':i in q11,'是否参与非Q11':i in non,'文献分组':'混合参与' if i in q11 and i in non else 'ITC参与组' if i in q11 else '纯非ITC组'})
    x=pd.DataFrame(rows).set_index('WOSID').join(meta,how='left').reset_index(); return x

def main():
    for p in [PAIR,AFILE,DFILE]:
        if not p.exists():raise FileNotFoundError(p)
    print('读取数据……'); am=prep(AFILE); dm=prep(DFILE)
    p=pd.read_csv(PAIR,usecols=['文献A_WOSID','文献D1_WOSID','文献D2_WOSID','quadrant'],dtype=str)
    for c in ['文献A_WOSID','文献D1_WOSID','文献D2_WOSID']:p[c]=nid(p[c])
    p['D低']=p[['文献D1_WOSID','文献D2_WOSID']].min(axis=1); p['D高']=p[['文献D1_WOSID','文献D2_WOSID']].max(axis=1); before=len(p); p=p.drop_duplicates(['文献A_WOSID','D低','D高','quadrant']).copy(); dup=before-len(p)
    print('构造文献层角色……'); adoc=doc_role(p,am,['文献A_WOSID'],'A'); ddoc=doc_role(p,dm,['文献D1_WOSID','文献D2_WOSID'],'D')
    doccmp=[]
    for role,x in [('A',adoc),('D',ddoc)]:
        exp=x[x.文献分组.eq('ITC参与组')]; pure=x[x.文献分组.eq('纯非ITC组')]; mixed=x[x.文献分组.eq('混合参与')]
        doccmp+=compare(exp,pure,f'{role}文献：ITC参与组 vs 纯非ITC组',RAW)
        doccmp+=compare(mixed,pure,f'{role}文献：混合参与组 vs 纯非ITC组',RAW)
    audit=pd.concat([adoc,ddoc]).groupby(['角色','文献分组']).size().rename('文献数').reset_index()
    print('构造配对层指标……')
    pair=p[['文献A_WOSID','文献D1_WOSID','文献D2_WOSID','quadrant']].copy()
    metrics=[]
    for c in RAW:
        pair['A_'+c]=pair['文献A_WOSID'].map(am[c]); pair['D1_'+c]=pair['文献D1_WOSID'].map(dm[c]); pair['D2_'+c]=pair['文献D2_WOSID'].map(dm[c])
        pair['D均值_'+c]=pair[['D1_'+c,'D2_'+c]].mean(axis=1); pair['D最大值_'+c]=pair[['D1_'+c,'D2_'+c]].max(axis=1); pair['D差值_'+c]=(pair['D1_'+c]-pair['D2_'+c]).abs(); metrics += ['A_'+c,'D均值_'+c,'D最大值_'+c,'D差值_'+c]
    pc=[]; q=pair[pair.quadrant.eq('Q11')]
    for g in GROUPS[1:]:pc+=compare(q,pair[pair.quadrant.eq(g)],f'配对层：Q11 vs {g}',metrics)
    # 同A 1:1匹配，控制A固有影响力差异
    mc=[]
    for g in GROUPS[1:]:
        qx=q.sample(frac=1,random_state=SEED).copy(); cx=pair[pair.quadrant.eq(g)].sample(frac=1,random_state=SEED+GROUPS.index(g)).copy(); qx['_r']=qx.groupby('文献A_WOSID').cumcount(); cx['_r']=cx.groupby('文献A_WOSID').cumcount(); m=qx.merge(cx,on=['文献A_WOSID','_r'],suffixes=('_Q11','_'+g))
        for metric in [x for x in metrics if x.startswith('D')]:
            b=m[[metric+'_Q11',metric+'_'+g]].dropna(); n1,n0,u,pv,d=effect(b.iloc[:,0],b.iloc[:,1]); mc.append({'比较':f'Q11 vs {g}','指标':metric,'匹配对数':len(b),'Q11覆盖率':len(m)/max(1,len(q)),'Q11中位数':b.iloc[:,0].median(),'控制组中位数':b.iloc[:,1].median(),'p值':pv,"Cliff's delta":d,'效应量等级':grade(d) if d==d else ''})
    cov=[]
    for g,x in pair.groupby('quadrant'):
        row={'组别':g,'无序三元组数':len(x)}
        for c in RAW:row[c+'三方完整率']=x[['A_'+c,'D1_'+c,'D2_'+c]].notna().all(axis=1).mean()
        cov.append(row)
    method=pd.DataFrame([['四组口径','完整配对主表Q11/Q10/Q01/Q00'],['文献层','ITC参与组、混合参与组、纯非ITC组分开，避免污染'],['配对层','A、D1/D2对称汇总，不给D1/D2强行指定B/C'],['稳健性','同一A内Q11与控制组1:1匹配'],['指标','使用次数180天、使用次数总、被引WOSCC、被引ALLDB'],['统计','Mann-Whitney U与Cliff’s delta'],['无序重复删除数',dup]],columns=['项目','说明'])
    with pd.ExcelWriter(OUT,engine='openpyxl') as w:
        method.to_excel(w,sheet_name='分析说明',index=False); audit.to_excel(w,sheet_name='文献参与类型审计',index=False); pd.DataFrame(doccmp).to_excel(w,sheet_name='文献层比较',index=False); pd.DataFrame(cov).to_excel(w,sheet_name='配对层覆盖率',index=False); pd.DataFrame(pc).to_excel(w,sheet_name='配对层四组比较',index=False); pd.DataFrame(mc).to_excel(w,sheet_name='同A匹配比较',index=False); adoc.to_excel(w,sheet_name='A文献分组明细',index=False); ddoc.to_excel(w,sheet_name='D文献分组明细',index=False); pair[pair.quadrant.eq('Q11')].head(10000).to_excel(w,sheet_name='Q11配对审计样本',index=False)
    print('完成：',OUT)
if __name__=='__main__':main()
