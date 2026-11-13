
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import *
def main():
    df=retained_data(); Y=df[PROPERTIES].to_numpy(float); n=len(df)
    all_rows=[]; summaries=[]
    for descriptor,degree in CANDIDATES:
        x=df[descriptor].to_numpy(float); model=fit(x,Y,degree); pred=predict(model,x)
        h=np.sum(model['q']**2,axis=1); cutoff=PLAN['leverage_multiplier']*(degree+1)/n
        assert np.isclose(h.sum(),degree+1,atol=1e-9) and np.all((h>=0)&(h<1))
        for j,prop in enumerate(PROPERTIES):
            residual=Y[:,j]-pred[:,j]; s=np.sqrt(np.sum(residual**2)/(n-degree-1))
            studentized=residual/(s*np.sqrt(1-h))
            high=h>cutoff; large=np.abs(studentized)>PLAN['residual_diagnostic_threshold']
            for i in range(n):
                all_rows.append({'Serial Number':int(df.iloc[i]['Serial Number']),'Name':df.iloc[i].Name,
                                 'Property':prop,'Descriptor':descriptor,'Order':degree,
                                 'Descriptor_value':x[i],'Actual':Y[i,j],'Fitted':pred[i,j],
                                 'Residual':residual[i],'Internally_studentized_residual':studentized[i],
                                 'Leverage':h[i],'Leverage_cutoff':cutoff,'High_leverage':bool(high[i]),
                                 'Large_residual_diagnostic':bool(large[i])})
            summaries.append({'Property':prop,'Descriptor':descriptor,'Order':degree,'N':n,'Parameters':degree+1,
                              'Leverage_cutoff':cutoff,'Maximum_leverage':float(h.max()),
                              'High_leverage_count':int(high.sum()),'Large_residual_count':int(large.sum()),
                              'Both_high_leverage_and_large_residual':int((high&large).sum()),
                              'High_leverage_serials':';'.join(map(str,df.loc[high,'Serial Number'].tolist())),
                              'Large_residual_serials':';'.join(map(str,df.loc[large,'Serial Number'].tolist()))})
    points=pd.DataFrame(all_rows)
    write_csv(points,'full_fit_applicability_diagnostics.csv')
    write_csv(pd.DataFrame(summaries),'full_fit_applicability_summary.csv')
    write_csv(points[(points.Descriptor=='DegEntropy')&(points.Order==3)&
                     (points.High_leverage|points.Large_residual_diagnostic)],
              'entropy_cubic_flagged_compounds.csv')
    ranges=[]
  
    for column in [*INDICES,'NumHeavyAtoms','NumBonds','MolWt']:
        values=df[column].to_numpy(float)
        ranges.append({'Quantity':column,'N':n,'Min':values.min(),'Q025':np.quantile(values,.025),
                       'Q25':np.quantile(values,.25),'Median':np.median(values),'Q75':np.quantile(values,.75),
                       'Q975':np.quantile(values,.975),'Max':values.max()})
    write_csv(pd.DataFrame(ranges),'observed_descriptor_and_size_ranges.csv')
    write_json({'leverage_rule':'h*=3p/n; p includes the intercept',
                'full_cubic_cutoff':PLAN['leverage_multiplier']*4/n,
                'residual_rule':'Internally studentized fitted-data residual; absolute value > 3 is diagnostic only',
                'data_scope':'All 221 retained compounds; fitted-data diagnostics only',
                'observed_ranges':'Descriptor, heavy-atom, bond-count, and molecular-weight ranges describe the available data',
                'interpretation':'Leverage and residual flags identify unusual fitted-data observations; they do not certify prediction reliability or broad chemical-space coverage',
                },
               RESULTS/'applicability_method.json')

if __name__=='__main__':
    prepare()
    with threadpool_limits(limits=1): main()
