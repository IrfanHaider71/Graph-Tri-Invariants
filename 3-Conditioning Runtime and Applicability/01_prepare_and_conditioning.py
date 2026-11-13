
import importlib.metadata
import os
import platform
import shutil
import sys
import numpy as np
import pandas as pd
import networkx as nx
from rdkit.Chem import Descriptors
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_info,threadpool_limits
from common import *

def main():
    prepare()
    for source,destination in [(Path(os.environ.get('QSPR_INPUT', str(SOURCE_ROOT/'1-Descriptor'/'inputs/dataset_combined.xlsx'))).expanduser().resolve(),INPUTS/'dataset_combined.xlsx'),
                               *[(SOURCE_ROOT/'1-Descriptor'/'generated'/'results_csv'/'analysis'/name,INPUTS/('baseline_'+name))
                                 for name in ['dataset_with_indices.csv','full_cubic_statistics.csv']]]:
        if destination.exists() and sha256(destination)!=sha256(source): raise AssertionError('Input snapshot differs')
        shutil.copyfile(source,destination)
    tracked=[p for p in SOURCE_ROOT.rglob('*')
             if p.is_file() and p.suffix in {'.py','.md'}
             and not any(part in {'generated','__pycache__','.git'} for part in p.relative_to(SOURCE_ROOT).parts)]
    tracked += [INPUTS/'dataset_combined.xlsx', INPUTS/'baseline_dataset_with_indices.csv', INPUTS/'baseline_full_cubic_statistics.csv']
    write_json({p.relative_to(SOURCE_ROOT).as_posix():sha256(p) for p in tracked},ENVIRONMENT/'original_file_hashes.json')
    original_count=verify_originals()
    source=pd.read_excel(INPUTS/'dataset_combined.xlsx')
    assert len(source)==PLAN['source_records_expected'] and source['Serial Number'].tolist()==list(range(1,224))
    baseline=pd.read_csv(INPUTS/'baseline_dataset_with_indices.csv',float_precision='round_trip')
    assert list(source.columns)==list(baseline.columns[:len(source.columns)])
    alignment=[]
    for column in source:
        if pd.api.types.is_numeric_dtype(source[column]):
            equal=np.isclose(source[column],baseline[column],rtol=1e-12,atol=1e-10,equal_nan=True)
        else: equal=source[column].fillna('<NULL>').eq(baseline[column].fillna('<NULL>')).to_numpy()
        assert np.all(equal),column
        alignment.append({'File':'baseline_dataset_with_indices.csv','Column':column,'Records_checked':len(source),'Matching_records':int(np.sum(equal))})
    write_csv(pd.DataFrame(alignment),'source_record_alignment.csv')

    records=[]; exclusion=[]
    with threadpool_limits(limits=1):
        for _,row in source.iterrows():
            graph,mol=mol_to_graph(row.SMILES)
            valid=graph is not None and graph.number_of_edges()>0 and nx.is_connected(graph)
            if not valid:
                reason='unparseable' if graph is None else 'edgeless' if graph.number_of_edges()==0 else 'disconnected'
                records.append({k:np.nan for k in [*INDICES,'NumHeavyAtoms','NumBonds','MolWt']})
                exclusion.append({'Serial Number':row['Serial Number'],'Name':row.Name,'Reason':reason,'SMILES':row.SMILES})
            else:
                records.append({**{key:function(graph) for key,function in DESCRIPTOR_FUNCTIONS.items()},
                                'NumHeavyAtoms':mol.GetNumHeavyAtoms(),'NumBonds':graph.number_of_edges(),
                                'MolWt':Descriptors.MolWt(mol)})
    recomputed=pd.concat([source.reset_index(drop=True),pd.DataFrame(records)],axis=1)
    write_csv(recomputed,'dataset_recomputed_all_223.csv')
    write_csv(pd.DataFrame(exclusion),'compound_exclusions.csv')
    df=recomputed.dropna(subset=INDICES).reset_index(drop=True)
    assert len(df)==PLAN['retained_records_expected'] and [r['Serial Number'] for r in exclusion]==[164,180]
    comparisons=[]
    for column in [*INDICES,'NumHeavyAtoms','NumBonds','MolWt']:
        a=recomputed[column].to_numpy(float); b=baseline[column].to_numpy(float)
        assert np.array_equal(np.isnan(a),np.isnan(b))
        good=np.isclose(a,b,rtol=1e-11,atol=1e-10,equal_nan=True)
        assert good.all(),column
        comparisons.append({'Quantity':column,'Rows_checked':len(source),'Matching_rows':int(good.sum()),'Maximum_absolute_difference':float(np.nanmax(np.abs(a-b)))})
    write_csv(pd.DataFrame(comparisons),'descriptor_reproduction_check.csv')

    conditioning=[]; full_rows=[]; prediction_comparison=[]
    y=df[PROPERTIES].to_numpy(float)
    for index,degree in CANDIDATES:
        x=df[index].to_numpy(float); X=design(x,degree); scales=np.linalg.norm(X,axis=0); Z=X/scales
        raw_beta,_,raw_rank,_=np.linalg.lstsq(X,y,rcond=None)
        model=fit(x,y,degree); prediction=predict(model,x); raw_prediction=X@raw_beta
        conditioning.append({'Descriptor':index,'Order':degree,'N':len(df),'Parameters':degree+1,
                             'Raw_condition_number_2norm':np.linalg.cond(X),'Scaled_condition_number_2norm':np.linalg.cond(Z),
                             'Condition_improvement_factor':np.linalg.cond(X)/np.linalg.cond(Z),
                             'Raw_rank_default_lstsq':int(raw_rank),'Scaled_rank':model['rank'],
                             **{f'L2_scale_column_{j}':float(scales[j]) for j in range(degree+1)}})
        h=np.sum(model['q']**2,axis=1); assert np.isclose(h.sum(),degree+1)
        for j,prop in enumerate(PROPERTIES):
            m=metrics(y[:,j],prediction[:,j]); raw=metrics(y[:,j],raw_prediction[:,j]); p=degree+1; residual=y[:,j]-prediction[:,j]
            sse=np.sum(residual**2); sst=np.sum((y[:,j]-y[:,j].mean())**2); f=((sst-sse)/degree)/(sse/(len(df)-p))
            full_rows.append({'Property':prop,'Descriptor':index,'Order':degree,'N':len(df),**m,
                              'Adjusted_R2':1-(1-m['R2'])*(len(df)-1)/(len(df)-p),
                              'SEE':np.sqrt(sse/(len(df)-p)),'F':f,'p_survival_function':float(f_distribution.sf(f,degree,len(df)-p)),
                              'Observed_fitted_R':float(np.corrcoef(y[:,j],prediction[:,j])[0,1]),
                              'Descriptor_property_r':float(np.corrcoef(x,y[:,j])[0,1]),
                              **{f'beta_{k}':float(model['beta'][k,j]) for k in range(degree+1)}})
            prediction_comparison.append({'Property':prop,'Descriptor':index,'Order':degree,'Raw_R2':raw['R2'],'Scaled_R2':m['R2'],
                                          'Raw_RMSE':raw['RMSE'],'Scaled_RMSE':m['RMSE'],
                                          'Maximum_prediction_difference':float(np.abs(raw_prediction[:,j]-prediction[:,j]).max())})
    write_csv(pd.DataFrame(conditioning),'conditioning_summary.csv')
    write_csv(pd.DataFrame(prediction_comparison),'conditioning_prediction_comparison.csv')
    full=pd.DataFrame(full_rows); write_csv(full,'full_fit_statistics_corrected_pvalues.csv')
    original_full=pd.read_csv(INPUTS/'baseline_full_cubic_statistics.csv')
    fit_checks=[]
    for _,row in original_full.iterrows():
        new=full[(full.Property==row.Property)&(full.Descriptor==row['Index'])&(full.Order==3)].iloc[0]
        assert np.isclose(new.R2,row.R2,atol=1e-11) and np.isclose(new.SEE,row.SEE,atol=1e-9)
        fit_checks.append({'Property':row.Property,'Descriptor':row['Index'],'Baseline_R2':row.R2,'Recomputed_R2':new.R2,
                           'Baseline_p':row.p,'Corrected_p':new.p_survival_function})
    write_csv(pd.DataFrame(fit_checks),'baseline_cubic_statistics_check.csv')
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','networkx','rdkit','scipy','matplotlib','openpyxl','threadpoolctl']}
    hardware={'cpu_model':platform.processor(),'physical_cores':None,'logical_processors':os.cpu_count(),'physical_memory_bytes':None}
    if os.name=='nt':
        import winreg
        import ctypes
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                hardware['cpu_model']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        except OSError: pass
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),
                      ('page_total',ctypes.c_ulonglong),('page_available',ctypes.c_ulonglong),('virtual_total',ctypes.c_ulonglong),
                      ('virtual_available',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
        memory=MemoryStatus();memory.length=ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):hardware['physical_memory_bytes']=memory.total
    try:
        import psutil
        hardware['physical_cores']=psutil.cpu_count(logical=False)
        hardware['physical_memory_bytes']=psutil.virtual_memory().total
    except ImportError: pass
    metadata={'analysis_date':PLAN['analysis_date'],'python':sys.version,
              'platform':platform.platform(),'machine':platform.machine(),'processor_identifier':os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),
              'logical_cpu_count':os.cpu_count(),**hardware,'package_versions':versions,'threadpools_during_preparation':[{**pool, 'filepath': Path(pool['filepath']).name} for pool in threadpool_info()],
              'thread_policy':'BLAS/OpenMP limited to one thread during fits and runtime benchmarks',
              'source_workbook_sha256':sha256(INPUTS/'dataset_combined.xlsx')}
    write_json(metadata,ENVIRONMENT/'runtime_environment.json')
    (ENVIRONMENT/'requirements-observed.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n',encoding='utf-8')
    assert verify_originals()==original_count
    print(f'Prepared {len(source)} records, retained {len(df)}; all descriptors and baseline cubic R2 values reproduced.',flush=True)
    print(conditioning and pd.DataFrame(conditioning)[['Descriptor','Order','Raw_condition_number_2norm','Scaled_condition_number_2norm']].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
