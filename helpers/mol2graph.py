import json
import numpy as np
import torch
import random

GLOBAL_NODES_FEATURES = {
    "atoms": ["mass", "radius", "pcharge", "x", "y", "z"],
    "angles": ["mass", "radius", "pcharge"],
    "dihedrals": ["mass", "radius", "pcharge"],
}

EDGE_FEATURES = [
    # "distance",
    "bonds",
    "van_der_waals",
    "coulomb",
]


def get_richgraph(path: str, bonds=False, rescale=False):
    with open(path, "r") as f:
        m = json.load(f)

    atoms = m["atoms"]
    # include atom_type properties
    for atom in atoms:
        atom["mass"] = m["atom_types"][atom["type"]]["mass"]
        atom["radius"] = m["atom_types"][atom["type"]]["radius"]

    # matrix represented as a list of dicts
    edges = []
    _ = [[edges.append({
        "atoms": [i, j],
        "bonds": m["bonds"][i][j],
        "van_der_waals": m["van_der_waals"][i][j],
        "coulomb": m["coulomb"][i][j],
        # "distance": m["distance"][i][j]
    }) for i, row in enumerate(m["bonds"])] for j, col in enumerate(m["bonds"])]

    angles = m["angles"]
    dihedrals = m["dihedrals"]

    return atoms, edges, angles, dihedrals


def get_nodes_features(atoms: list, node_type: str):
    matrix = np.zeros(shape=(len(atoms), len(GLOBAL_NODES_FEATURES[node_type])))  # x [num_nodes, num_node_features]

    for i, atom in enumerate(atoms):
        for j, feat in enumerate(GLOBAL_NODES_FEATURES[node_type]):
            matrix[i, j] = atom[feat]

    # To be normalized?

    return torch.from_numpy(matrix).float()


def get_edges_interactions(edges_interactions: list):
    edge_index = np.zeros(shape=(2, len(edges_interactions)))  # In the edge_index format [2, num_edges]
    matrix = np.zeros(shape=(len(edges_interactions), len(EDGE_FEATURES)))  # In the edge_attr format [num_edges, num_edge_features]

    for i, edge in enumerate(edges_interactions):
        edge_index[:, i] = np.asarray(edge["atoms"])
        for j, feature in enumerate(EDGE_FEATURES):
            matrix[i, j] = edge[feature]

    # To be normalized?

    return torch.from_numpy(edge_index).long(), torch.from_numpy(matrix).float()


def get_atoms_interactions_graph(atoms, edges):
    # X: node_features matrix
    node_features = get_nodes_features(atoms, "atoms")

    # [2, num_edges], [num_edges, num_edge_features]
    edge_index, edge_features = get_edges_interactions(edges)

    return node_features, edge_index, edge_features


def get_angles_graph(atoms, angles):

    # First we have to compute different nodes (since in the angles_graph nodes are couple of atoms)
    # TODO: Do they have features? My guess is that angle value between different couple of atoms impacts in a different
    # TODO: way on the target free energy.

    return get_graph(angles, atoms, "angles")


def get_graph(angles: list, atoms_list: list, angle_type: str, angles_list=None):
    """
        N.B: The following assumptions has to be made for dihedrals graph too.

        In the angles graph we want to represent an angle as a relationship between two couples of atoms: if I have 3 atoms,
        formerly 0, 1 and 2, an angle in 1 of 120° I want to draw a graph like this: (0,1) --120°-> (1,2)

        Here some questions may raise:
        + Should we represent an angle as a bi-directional relationship? (making the edge undirected)
            - It means to add an edge like this: (1,2) --120°-> (0,1)
              Physically speaking, this doesn't hold since the angle between these two vectors is the explementary of 120°,
              so it should be 240°.
            - Now the question could be: should the neural net be aware of this? should we add this information?
              If we add no information it means that there's no relationship between the two couples in the other way round,
              but it's wrong, since they hold the explementary angle.

        + The vertex (0,1) is the same as (1,0)?
            - It depends on how we treat the angles.
              One simplification we can do is to use the angle relationship as a bidirectional one but without considering the
              vertex order (atoms inside the vertex). so we map (0,1) and (1,0) as the same couple A; the same is done
              for (1,2) and (2,1) as B and we say we have a relationship (A) <--120°--> (B)

            - Another point of view could be to see what differs in reality:
              (Try to expand this when you have time)


        Our assumptions:
        + Angle is a bidirectional relationship: (0,1) <--x--> (1,2)
        + Order in nodes doesn't matter: (0,1) is the same as (1,0) and (0,1) <--x--> (1,2) is the same as (1,0) <--x--> (1,2)
          or again as (1,0) <--x--> (2,1).

        """
    nodes_tuple = set()
    for i, angle in enumerate(angles):
        # Sort is done in order to don't take order into account
        first = sorted(angle["atoms"][:len(angle["atoms"]) - 1])
        second = sorted(angle["atoms"][1:])
        nodes_tuple.add(tuple(first))
        nodes_tuple.add(tuple(second))

    nodes = list(nodes_tuple)

    map_to_index = {k: v for v, k in enumerate(nodes)}
    # Each angle has two edges (since it's bidirectional)
    edge_index = np.zeros(shape=(2, 2 * len(angles)), dtype=np.long)
    edge_features = np.zeros(shape=(2 * len(angles), 1))
    for i, angle in enumerate(angles):
        # Sort is done in order to don't take order into account
        first = sorted(angle["atoms"][:len(angle["atoms"]) - 1])
        second = sorted(angle["atoms"][1:])
        edge_index[0, i] = map_to_index[tuple(first)]
        edge_index[1, i] = map_to_index[tuple(second)]
        edge_index[0, i + len(angles)] = map_to_index[tuple(second)]
        edge_index[1, i + len(angles)] = map_to_index[tuple(first)]
        value = float(angle["value"])
        edge_features[i, 0] = value
        edge_features[i + len(angles), 0] = value

    # TODO: Understand if nodes in this graph have features or not. For now give them at least the sum of the masses
    nodes_features = np.zeros(shape=(len(nodes), len(GLOBAL_NODES_FEATURES[angle_type])))
    for i, node in enumerate(nodes):
        nodes_features[i] = [
           sum_properties(list(node), atoms_list, prop) for prop in GLOBAL_NODES_FEATURES[angle_type]
        ]

    return (
        torch.from_numpy(nodes_features).float(),
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_features).float()
    )


def sum_properties(atoms_indexes, atoms_list: list, prop: str):
    s = 0
    for atom in atoms_indexes:
        s += float(atoms_list[atom][prop])
    return s


def get_dihedrals_graph(atoms, dihedrals, angles_list):
    # First we have to compute different nodes (since in the angles_graph nodes are couple of atoms)
    # TODO: Do they have features? My guess is that angle value between different couple of atoms impacts in a different
    # TODO: way on the target free energy.

    return get_graph(dihedrals, atoms, "dihedrals", angles_list)


def get_debruijn_graph(atoms, angles, dihedrals, shuffle=False):
    # maps to get directly value from angles and dihedrals
    if shuffle:
        random.shuffle(dihedrals)
    atoms_to_dihedral = {
        tuple(sorted(dihedral["atoms"])): dihedral["value"] for dihedral in dihedrals
    }
    atoms_to_angle = {
        tuple(sorted(angle["atoms"])): angle["value"] for angle in angles
    }

    # Build an overlap graph
    overlap_nodes = []
    for dihedral in dihedrals:
        overlap_nodes.append(sorted(dihedral["atoms"]))

    edges = []
    for i, dihedral1 in enumerate(overlap_nodes):
        for j, dihedral2 in enumerate(overlap_nodes):
            # If they have 3 atoms in common, then create the edge
            atoms_in_common = [i for i in dihedral1 if i in dihedral2]
            if len(atoms_in_common) == 3:
                edges.append([i, j])
                # TODO: put a value on the edge here
                # angle_to_value(tuple(sorted(atoms_in_common)))

    # TODO: (for now there aren't) Remove nodes without an edge
    # There are helpers for this

    nodes_features = np.zeros(shape=(len(overlap_nodes), 1))
    for i, node in enumerate(overlap_nodes):
        # nodes_features[i] = [atoms_to_dihedral[tuple(node)], sum_properties(node, atoms, "mass")]
        nodes_features[i] = [atoms_to_dihedral[tuple(node)]]

    edge_index = np.asarray(edges).transpose()

    return (
        torch.from_numpy(nodes_features).float(),
        torch.from_numpy(edge_index).long(),
    )
