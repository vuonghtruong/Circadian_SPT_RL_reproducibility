"""Export the two manuscript tables at full numerical precision."""
from common import *

def main():
    g=pd.read_csv(OUT/'results/group_cosinor.csv').set_index('group')
    p=pd.read_csv(OUT/'results/figure2_phase_means_sem.csv').set_index(['Sex','age_group'])
    one=[];two=[]
    for sex in ['M','F']:
        for age in ['Young','EarlyMid','Mid','Old']:
            r=g.loc[sex+' '+age];q=p.loc[(sex,age)]
            one.append(dict(Sex=sex,age_group=age,n_sessions=int(r.n_sessions),mesor=r.group_mean_mesor,
              amplitude=r.group_mean_amplitude,acrophase_h=r.group_mean_phase_h,cosinor_p=r.p_group_mean_OLS,
              cosinor_BH_q=r.q_group_mean_OLS,emergence_index=r.emergence,permutation_p=r.p_phase_permutation,
              permutation_BH_q=r.q_phase_permutation))
            two.append(dict(Sex=sex,age_group=age,**q.to_dict()))
    (OUT/'tables').mkdir(exist_ok=True)
    pd.DataFrame(one).to_csv(OUT/'tables/Table1_group_cosinor.csv',index=False,lineterminator='\n')
    pd.DataFrame(two).to_csv(OUT/'tables/Table2_acrophase_means_SEM.csv',index=False,lineterminator='\n')
    print('Exported Tables 1 and 2')

if __name__=='__main__':main()
