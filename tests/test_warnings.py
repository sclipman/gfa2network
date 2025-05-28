import networkx as nx
import pytest
from gfa2network.analysis import genome_distance


def test_mean_distance_warning():
    G = nx.Graph()
    G.add_node("hub")
    for i in range(50):
        node = f"a{i}"
        G.add_edge("hub", node)
    for i in range(21):
        node = f"b{i}"
        G.add_edge("hub", node)
    set_a = [f"a{i}" for i in range(50)]
    set_b = [f"b{i}" for i in range(21)]
    with pytest.warns(RuntimeWarning):
        genome_distance(G, set_a, set_b, method="mean")
