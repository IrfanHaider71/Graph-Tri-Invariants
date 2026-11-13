
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors

from _paths import INPUT, READ_RESULTS, OUTPUT_RESULTS, OUTPUT_FIGURES, prepare_outputs

SRC = str(INPUT)
OUT_CSV = str(OUTPUT_RESULTS / "dataset_with_indices.csv")


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G, mol


def graph_energy(G):
    A = nx.to_numpy_array(G)
    eigvals = np.linalg.eigvalsh(A)
    return float(np.sum(np.abs(eigvals)))


def schultz_index(G):
    dist = dict(nx.all_pairs_shortest_path_length(G))
    deg = dict(G.degree())
    total = 0.0
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            total += (deg[u] + deg[v]) * dist[u][v]
    return total


def degree_entropy(G):
    deg = np.array([d for _, d in G.degree()], dtype=float)
    two_m = deg.sum()
    p = deg / two_m
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_all(smiles):
    G, mol = mol_to_graph(smiles)
    if G is None or G.number_of_edges() == 0:
        return None
    if not nx.is_connected(G):
        return None 
    return {
        "GraphEnergy": graph_energy(G),
        "Schultz": schultz_index(G),
        "DegEntropy": degree_entropy(G),
        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "NumBonds": G.number_of_edges(),
        "MolWt": Descriptors.MolWt(mol),
    }


def main():
    df = pd.read_excel(SRC)
    print(f"Loaded {len(df)} rows")

    records, failures = [], []
    for _, row in df.iterrows():
        res = compute_all(row["SMILES"])
        if res is None:
            failures.append((row["Serial Number"], row["Name"], row["SMILES"]))
            records.append({k: None for k in ["GraphEnergy", "Schultz", "DegEntropy", "NumHeavyAtoms", "NumBonds", "MolWt"]})
        else:
            records.append(res)

    idx_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), idx_df], axis=1)

    print(f"Failures: {len(failures)}")
    for f in failures:
        print("  ", f)
    print(f"Missing values:\n{out.isna().sum()}")

    out.to_csv(OUT_CSV, index=False)

    print("\nDescriptive stats:")
    cols = ["GraphEnergy", "Schultz", "DegEntropy", "Boiling Point (BP)", "Molar Volume (MV)",
            "Flash Point (FP)", "Polarizability (Pol)", "Molar Refractivity (MR)"]
    print(out.dropna(subset=["GraphEnergy", "Schultz", "DegEntropy"])[cols].describe().T.round(3))

    print("\nCorrelation among the 3 new indices:")
    print(out[["GraphEnergy", "Schultz", "DegEntropy"]].corr().round(4))


if __name__ == "__main__":
    prepare_outputs()
    main()
