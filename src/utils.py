from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def plot_dag(dag: nx.DiGraph):
    """
    Utility function to quickly draw a Directed Acyclic Graph.
    Root is at the top, leaves are on the bottom.
    Args:
        dag: input directed acyclic graph
    """
    pos = graphviz_layout(dag, prog="dot")
    nx.draw(dag, pos, with_labels=False, node_size=10, arrows=False)
    plt.show()
