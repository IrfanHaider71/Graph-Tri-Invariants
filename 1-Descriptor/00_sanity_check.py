
import numpy as np
import networkx as nx
from rdkit import Chem

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


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
    return float(-np.sum(p * np.log2(p)))


if __name__ == "__main__":
    G = mol_to_graph("C1CCC2=NC3=CC=CC=C3C(=C2C1)N")  # butane
    e = graph_energy(G)
    sc = schultz_index(G)
    ent = degree_entropy(G)
    print(f"Graph energy: {e:.4f}  (expected ~4.4721)")
    print(f"Schultz index: {sc:.4f}  (expected 28.0000)")
    print(f"Degree entropy: {ent:.4f}  (expected ~1.9183)")
