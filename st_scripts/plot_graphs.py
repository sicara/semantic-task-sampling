#%%

#%%
from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from src.easyfsl import EasySemantics, EasySet
from src.easyfsl import DanishFungi

#%%

dataset = "tiered_imagenet"


if dataset == "fungi":
    easy_set = DanishFungi()
    semantic_tools = EasySemantics(
        easy_set, Path("data/fungi/specs/fungi_dag.json"), is_fungi=True
    )
else:
    easy_set = EasySet(specs_file="data/tiered_imagenet/specs/test.json", training=False)
    semantic_tools = EasySemantics(easy_set, Path("data/tiered_imagenet/specs") / "wordnet.is_a.txt")

G = semantic_tools.dataset_dag

#%%
# pos = graphviz_layout(G, prog="twopi", root="root")
pos = graphviz_layout(G, prog="twopi", root="n00001740")

#%%
colors = []
sizes = []
for node in G:
    # if node == "root":
    if node == "n00001740":
        colors.append("green")
        sizes.append(22)
    elif node == 'Protozoa':
        colors.append("violet")
        sizes.append(15)
    elif node == 'Chromista':
        colors.append("cyan")
        sizes.append(8)
    elif node == 'Fungi':
        colors.append("orange")
        sizes.append(10)
    elif G.out_degree(node) == 0 :
        colors.append("red")
        sizes.append(14)
    else:
        colors.append("blue")
        sizes.append(10)

nx.draw(G, pos, with_labels=False, node_size=sizes,
        arrows=True, arrowstyle="->", arrowsize= 5,
        # arrows=False, width=0.05,
        node_color= colors,
        )
# plt.savefig("fungi_tree.svg")
plt.savefig("tiered_graph.pdf")
