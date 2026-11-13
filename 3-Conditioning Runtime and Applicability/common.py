
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent
RESULTS = ROOT / 'generated' / 'results_csv'
FIGURES = ROOT / 'generated' / 'figures'
INPUTS = ROOT / 'inputs'
ENVIRONMENT = ROOT / 'generated' / 'environment'
PLAN = {'analysis_date': '2026-11-05', 'source_records_expected': 223, 'retained_records_expected': 221, 'seed': 42, 'benchmark_rounds': 30, 'benchmark_warmup_rounds': 3, 'benchmark_compound_repeats': 7, 'benchmark_compound_inner_loops': 5, 'benchmark_blas_threads': 1, 'leverage_multiplier': 3, 'residual_diagnostic_threshold': 3}
INDICES = ['GraphEnergy', 'Schultz', 'DegEntropy']
PROPERTIES = ['Boiling Point (BP)', 'Flash Point (FP)', 'Molar Volume (MV)',
              'Polarizability (Pol)', 'Molar Refractivity (MR)']
SHORT = dict(zip(PROPERTIES, ['BP','FP','MV','Pol','MR']))
CANDIDATES = [(index,degree) for index in INDICES for degree in [1,2,3]]


def prepare():
    for directory in [RESULTS,FIGURES,INPUTS,ENVIRONMENT]: directory.mkdir(parents=True,exist_ok=True)


def write_csv(frame,name):
    frame.to_csv(RESULTS/name,index=False,encoding='utf-8-sig',float_format='%.17g')


def write_json(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mol_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None,None
    graph=nx.Graph()
    for atom in mol.GetAtoms(): graph.add_node(atom.GetIdx())
    for bond in mol.GetBonds(): graph.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
    return graph,mol


def graph_energy(graph):
    adjacency=nx.to_numpy_array(graph)
    return float(np.abs(np.linalg.eigvalsh(adjacency)).sum())


def schultz_index(graph):
    distances=dict(nx.all_pairs_shortest_path_length(graph))
    degrees=dict(graph.degree())
    total=0.0
    nodes=list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            u,v=nodes[i],nodes[j]
            total+=(degrees[u]+degrees[v])*distances[u][v]
    return total


def degree_entropy(graph):
    degrees=np.array([d for _,d in graph.degree()],dtype=float)
    probabilities=degrees/degrees.sum()
    probabilities=probabilities[probabilities>0]
    return float(-(probabilities*np.log2(probabilities)).sum())


DESCRIPTOR_FUNCTIONS=dict(zip(INDICES,[graph_energy,schultz_index,degree_entropy]))


def retained_data():
    return pd.read_csv(RESULTS/'dataset_recomputed_all_223.csv',float_precision='round_trip').dropna(subset=INDICES).reset_index(drop=True)


def design(x,degree):
    return np.vander(np.asarray(x,dtype=float),N=degree+1,increasing=True)


def fit(x,y,degree):
    matrix=design(x,degree)
    scales=np.linalg.norm(matrix,axis=0)
    scaled=matrix/scales
    beta_scaled,_,rank,singular=np.linalg.lstsq(scaled,y,rcond=None)
    if rank!=degree+1: raise ArithmeticError('Rank deficient scaled polynomial fit')
    beta=beta_scaled/scales[:,None] if np.ndim(y)==2 else beta_scaled/scales
    q,r=np.linalg.qr(scaled,mode='reduced')
    return {'beta':beta,'scales':scales,'q':q,'r':r,'degree':degree,
            'n':len(x),'minimum':float(np.min(x)),'maximum':float(np.max(x)),
            'rank':int(rank),'singular':singular}


def predict(model,x):
    return design(x,model['degree'])@model['beta']


def metrics(y,prediction):
    y=np.asarray(y); prediction=np.asarray(prediction)
    error=y-prediction
    sse=float(np.sum(error*error)); sst=float(np.sum((y-y.mean())**2))
    return {'R2':1-sse/sst if sst>0 and len(y)>1 else np.nan,
            'RMSE':float(np.sqrt(np.mean(error*error))), 'MAE':float(np.mean(np.abs(error)))}


def verify_originals():
    manifest=json.loads((ENVIRONMENT/'original_file_hashes.json').read_text(encoding='utf-8'))
    for relative,expected in manifest.items():
        if sha256(SOURCE_ROOT/relative)!=expected: raise AssertionError(f'Original file changed: {relative}')
    return len(manifest)
