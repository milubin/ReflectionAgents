import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

ROLE_COLORS = {
    "planner":     "#4A90D9",
    "researcher":  "#5CB85C",
    "analyst":     "#5CB85C",
    "historian":   "#5CB85C",
    "worker":      "#5CB85C",
    "critic":      "#D9534F",
    "synthesizer": "#F0AD4E",
    "input":       "#AAAAAA",
    "identity":    "#CCCCCC",
    "square":      "#9B59B6",
    "double":      "#9B59B6",
    "increment":   "#9B59B6",
    "negate":      "#9B59B6",
}

LEGEND = [
    mpatches.Patch(color="#4A90D9", label="Planner"),
    mpatches.Patch(color="#5CB85C", label="Worker / Analyst / Researcher"),
    mpatches.Patch(color="#D9534F", label="Critic"),
    mpatches.Patch(color="#F0AD4E", label="Synthesizer"),
    mpatches.Patch(color="#9B59B6", label="Transform node"),
    mpatches.Patch(color="#AAAAAA", label="Input / Identity"),
]


def _node_color(node_name: str) -> str:
    lower = node_name.lower()
    for key, color in ROLE_COLORS.items():
        if key in lower:
            return color
    return "#AED6F1"


def visualize_graph(
    G: nx.DiGraph,
    title: str = "Agent Graph",
    output_path: str = "agent_graph.png",
    layout: str = "spring",
) -> str:
    """
    Draw a networkx DiGraph and save it as a PNG image.

    Node colors are automatically assigned based on node name / role.
    Returns the output path.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    if layout == "hierarchical":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.5)

    colors = [_node_color(str(n)) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=2400, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#444444",
        arrows=True,
        arrowsize=22,
        width=2,
        connectionstyle="arc3,rad=0.1",
    )

    ax.legend(handles=LEGEND, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Graph visualization saved → {output_path}")
    return output_path
