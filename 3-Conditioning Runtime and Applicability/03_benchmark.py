
import gc
import random
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits,threadpool_info
from common import *
def measure(function,inner=1):
    was_enabled=gc.isenabled()
    gc.disable()
    try:
        start=time.perf_counter_ns()
        for _ in range(inner):function()
        elapsed=time.perf_counter_ns()-start
    finally:
        if was_enabled:gc.enable()
    return elapsed/inner

def main():
    df=retained_data(); smiles=df.SMILES.tolist(); graphs=[mol_to_graph(s)[0] for s in smiles]
    n=len(graphs); rng=random.Random(PLAN['seed'])
    def evaluate_all(descriptor,end_to_end=False):
        function=DESCRIPTOR_FUNCTIONS[descriptor]
        if end_to_end:
            return sum(function(mol_to_graph(s)[0]) for s in smiles)
        return sum(function(graph) for graph in graphs)
    for _ in range(PLAN['benchmark_warmup_rounds']):
        for descriptor in INDICES:evaluate_all(descriptor)
    timings=[]
    for mode in ['descriptor_only_prebuilt_graphs','end_to_end_parse_graph_descriptor']:
        for repeat in range(PLAN['benchmark_rounds']):
            order=INDICES.copy();rng.shuffle(order)
            for position,descriptor in enumerate(order):
                elapsed=measure(lambda:evaluate_all(descriptor,mode.startswith('end_to_end')))
                timings.append({'Mode':mode,'Round':repeat+1,'Execution_position':position+1,'Descriptor':descriptor,
                                'Molecules':n,'Dataset_time_ms':elapsed/1e6,'Per_molecule_time_us':elapsed/n/1e3})
            if (repeat+1)%10==0:print(f'{mode}: completed {repeat+1}/{PLAN["benchmark_rounds"]} rounds',flush=True)
    frame=pd.DataFrame(timings);write_csv(frame,'runtime_dataset_measurements.csv')
    summaries=[]
    for (mode,descriptor),part in frame.groupby(['Mode','Descriptor'],sort=False):
        values=part.Dataset_time_ms.to_numpy(float)
        summaries.append({'Mode':mode,'Descriptor':descriptor,'Molecules':n,'Rounds':len(part),
                          'Median_dataset_ms':np.median(values),'Mean_dataset_ms':np.mean(values),'SD_dataset_ms':np.std(values,ddof=1),
                          'Q25_dataset_ms':np.quantile(values,.25),'Q75_dataset_ms':np.quantile(values,.75),
                          'Q05_dataset_ms':np.quantile(values,.05),'Q95_dataset_ms':np.quantile(values,.95),
                          'Minimum_dataset_ms':values.min(),'Maximum_dataset_ms':values.max(),
                          'Median_per_molecule_us':np.median(values)*1000/n})
    write_csv(pd.DataFrame(summaries),'runtime_dataset_summary.csv')
    compound=[]
    for i,graph in enumerate(graphs):
        order=INDICES.copy();rng.shuffle(order)
        for descriptor in order:
            function=DESCRIPTOR_FUNCTIONS[descriptor]
            function(graph)
            values=[measure(lambda:function(graph),inner=PLAN['benchmark_compound_inner_loops'])/1e3
                    for _ in range(PLAN['benchmark_compound_repeats'])]
            compound.append({'Serial Number':int(df.iloc[i]['Serial Number']),'Name':df.iloc[i].Name,
                             'Descriptor':descriptor,'NumHeavyAtoms':graph.number_of_nodes(),'NumBonds':graph.number_of_edges(),
                             'Median_time_us':np.median(values),'Q25_time_us':np.quantile(values,.25),'Q75_time_us':np.quantile(values,.75),
                             'Timing_repeats':len(values),'Inner_loops':PLAN['benchmark_compound_inner_loops']})
    write_csv(pd.DataFrame(compound),'runtime_per_compound.csv')
    write_json({'protocol':'Original descriptor algorithms, randomized descriptor order each dataset round; no competing analysis processes launched',
                'timer':'time.perf_counter_ns','timer_resolution_seconds':time.get_clock_info('perf_counter').resolution,
                'warmup_passes':PLAN['benchmark_warmup_rounds'],'dataset_rounds_per_mode':PLAN['benchmark_rounds'],
                'gc':'disabled only within timed intervals; original GC state restored afterward',
                'thread_limit':1,'threadpools_during_benchmark':[{**pool, 'filepath': Path(pool['filepath']).name} for pool in threadpool_info()],
                'descriptor_only_scope':'descriptor calculation including matrix/BFS/degrees construction, with molecular graphs prebuilt',
                'end_to_end_scope':'RDKit SMILES parsing, NetworkX graph construction, and one descriptor; excludes Excel/CSV I/O and regression',
                'compound_scope':'prebuilt graph, seven timing repeats of five calls each; median reported',
                'limits':'Hardware/implementation-specific measurements on these 221 drug-sized graphs; background operating-system activity is not controlled and no large-graph scaling claim is made'},
               RESULTS/'runtime_protocol.json')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print('Benchmark complete. Original files unchanged:',verify_originals(),flush=True)

if __name__=='__main__':
    prepare()
    with threadpool_limits(limits=1):main()
