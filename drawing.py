from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx.drawing.nx_pydot import write_dot


def save_graph(nodes, edges, filename, to_undirected=True):
    graph = Data(x=nodes, edge_index=edges)
    nx_graph = to_networkx(graph, node_attrs=["x"], to_undirected=to_undirected)
    write_dot(nx_graph, filename)
    # dot_graph = to_pydot(nx_graph)
    # dot_graph.write(filename)
    pass
