
import ast
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits
from common import *

def main():
    source=pd.read_excel(INPUTS/'dataset_combined.xlsx')
    recorded=pd.read_csv(RESULTS/'dataset_recomputed_all_223.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(source,recorded[source.columns],check_dtype=False,check_exact=True)
    assert len(source)==PLAN['source_records_expected']
    assert source['Serial Number'].tolist()==list(range(1,len(source)+1))
    df=retained_data(); Y=df[PROPERTIES].to_numpy(float); n=len(df)
    assert n==PLAN['retained_records_expected']
    excluded=pd.read_csv(RESULTS/'compound_exclusions.csv')
    assert excluded['Serial Number'].tolist()==[164,180]
    assert (excluded.Reason=='disconnected').all()
    baseline=pd.read_csv(INPUTS/'baseline_dataset_with_indices.csv',float_precision='round_trip')
    for quantity in [*INDICES,'NumHeavyAtoms','NumBonds','MolWt']:
        assert np.allclose(recorded[quantity],baseline[quantity],rtol=1e-11,atol=1e-10,equal_nan=True)
    checks=[{'Check':'All 223 source records, original order, extracted workbooks, descriptors, and two exclusions match','Passed':True}]

    condition=pd.read_csv(RESULTS/'conditioning_summary.csv',float_precision='round_trip')
    full=pd.read_csv(RESULTS/'full_fit_statistics_corrected_pvalues.csv',float_precision='round_trip')
    diagnostics=pd.read_csv(RESULTS/'full_fit_applicability_diagnostics.csv',float_precision='round_trip')
    summary=pd.read_csv(RESULTS/'full_fit_applicability_summary.csv')
    assert len(condition)==9 and len(full)==45 and len(diagnostics)==n*45 and len(summary)==45
    assert not diagnostics.duplicated(['Serial Number','Property','Descriptor','Order']).any()
    max_prediction_difference=0.; max_leverage_difference=0.
    for descriptor,degree in CANDIDATES:
        x=df[descriptor].to_numpy(float); X=design(x,degree); Z=X/np.linalg.norm(X,axis=0)
        row=condition[(condition.Descriptor==descriptor)&(condition.Order==degree)].iloc[0]
        raw_s=np.linalg.svd(X,compute_uv=False); scaled_s=np.linalg.svd(Z,compute_uv=False)
        assert np.isclose(row.Raw_condition_number_2norm,raw_s[0]/raw_s[-1],rtol=1e-12)
        assert np.isclose(row.Scaled_condition_number_2norm,scaled_s[0]/scaled_s[-1],rtol=1e-12)
       
        centered=(x-x.mean())/x.std()
        Q=np.vander(centered,N=degree+1,increasing=True)
        expected=Q@np.linalg.lstsq(Q,Y,rcond=None)[0]
        u,_,_=np.linalg.svd(Z,full_matrices=False); h=np.sum(u*u,axis=1)
        assert np.isclose(h.sum(),degree+1)
        cutoff=PLAN['leverage_multiplier']*(degree+1)/n
        for j,prop in enumerate(PROPERTIES):
            part=diagnostics[(diagnostics.Descriptor==descriptor)&(diagnostics.Order==degree)&
                             (diagnostics.Property==prop)].reset_index(drop=True)
            assert part['Serial Number'].tolist()==df['Serial Number'].tolist()
            assert part.Name.tolist()==df.Name.tolist()
            assert np.array_equal(part.Actual.to_numpy(),Y[:,j])
            assert np.allclose(part.Fitted,expected[:,j],rtol=1e-9,atol=1e-7)
            assert np.allclose(part.Leverage,h,rtol=1e-9,atol=1e-10)
            max_prediction_difference=max(max_prediction_difference,float(np.abs(part.Fitted-expected[:,j]).max()))
            max_leverage_difference=max(max_leverage_difference,float(np.abs(part.Leverage-h).max()))
            residual=Y[:,j]-part.Fitted.to_numpy()
            sse=np.sum(residual**2); see=np.sqrt(sse/(n-degree-1))
            studentized=residual/(see*np.sqrt(1-h))
            assert np.allclose(part.Residual,residual,atol=1e-9)
            assert np.allclose(part.Internally_studentized_residual,studentized,rtol=1e-9,atol=1e-9)
            assert np.allclose(part.Leverage_cutoff,cutoff)
            assert np.array_equal(part.High_leverage,h>cutoff)
            assert np.array_equal(part.Large_residual_diagnostic,np.abs(studentized)>PLAN['residual_diagnostic_threshold'])
            stat=full[(full.Descriptor==descriptor)&(full.Order==degree)&(full.Property==prop)].iloc[0]
            sst=np.sum((Y[:,j]-Y[:,j].mean())**2)
            F=((sst-sse)/degree)/(sse/(n-degree-1))
            assert np.isclose(stat.R2,1-sse/sst,rtol=1e-9)
            assert np.isclose(stat.p_survival_function,f_distribution.sf(F,degree,n-degree-1),rtol=1e-8,atol=0)
            s=summary[(summary.Descriptor==descriptor)&(summary.Order==degree)&(summary.Property==prop)].iloc[0]
            assert s.High_leverage_count==int(part.High_leverage.sum())
            assert s.Large_residual_count==int(part.Large_residual_diagnostic.sum())
    assert ((full.p_survival_function>0)&(full.p_survival_function<1)).all()
    assert np.allclose(full.Observed_fitted_R**2,full.R2,atol=1e-9)
    original_fit=pd.read_csv(INPUTS/'baseline_full_cubic_statistics.csv')
    for row in original_fit.itertuples(index=False):
        new=full[(full.Property==row.Property)&(full.Descriptor==row.Index)&(full.Order==3)].iloc[0]
        assert np.isclose(new.R2,row.R2,rtol=1e-10,atol=1e-11)
        assert np.isclose(new.SEE,row.SEE,rtol=1e-10,atol=1e-9)
    checks.extend([
        {'Check':'Nine raw/scaled condition numbers and all 45 full-data fits checked; original 15 cubic fits preserved','Passed':True,
         'Maximum_fitted_value_difference':max_prediction_difference},
        {'Check':'All 9,945 fitted-data applicability records checked against independent SVD leverage and residual formulas','Passed':True,
         'Maximum_leverage_difference':max_leverage_difference}])

    timings=pd.read_csv(RESULTS/'runtime_dataset_measurements.csv',float_precision='round_trip')
    timing_summary=pd.read_csv(RESULTS/'runtime_dataset_summary.csv',float_precision='round_trip')
    assert len(timings)==2*PLAN['benchmark_rounds']*len(INDICES)
    assert (timings.Dataset_time_ms>0).all() and (timings.Molecules==n).all()
    for (mode,descriptor),part in timings.groupby(['Mode','Descriptor']):
        assert len(part)==PLAN['benchmark_rounds'] and part.Round.nunique()==PLAN['benchmark_rounds']
        result=timing_summary[(timing_summary.Mode==mode)&(timing_summary.Descriptor==descriptor)].iloc[0]
        for field,value in [('Median_dataset_ms',np.median(part.Dataset_time_ms)),
                            ('Q25_dataset_ms',np.quantile(part.Dataset_time_ms,.25)),
                            ('Q75_dataset_ms',np.quantile(part.Dataset_time_ms,.75))]:
            assert np.isclose(result[field],value,rtol=1e-12)
    compound=pd.read_csv(RESULTS/'runtime_per_compound.csv')
    assert len(compound)==n*len(INDICES) and (compound.Median_time_us>0).all()
    assert not compound.duplicated(['Serial Number','Descriptor']).any()
    checks.append({'Check':'180 dataset timing measurements, six summaries, and 663 per-compound timing records consistent','Passed':True})
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    count=verify_originals()
    checks.append({'Check':f'All scripts parse and all {count} original project files remain byte-identical','Passed':True})
    write_json({'all_checks_passed':True,'checks':checks,'source_records':len(source),'retained_records':n,
                'external_validation_performed':False},RESULTS/'verification.json')
    for check in checks:print('PASS:',check,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
