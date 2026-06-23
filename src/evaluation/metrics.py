def compute_shd(true_edges: set, learned_edges: set, all_nodes: list) -> int:
    """Symmetric difference of true and learned edge sets."""
    return len(true_edges.symmetric_difference(learned_edges))


def compute_fp_rate(true_edges: set, learned_edges: set, all_nodes: list) -> float:
    """Proportion of learned edges not in the true graph."""
    if len(learned_edges) == 0:
        return 0.0
    return len(learned_edges - true_edges) / len(learned_edges)


def compute_fn_rate(true_edges: set, learned_edges: set, all_nodes: list) -> float:
    """Proportion of true edges not recovered."""
    if len(true_edges) == 0:
        return 0.0
    return len(true_edges - learned_edges) / len(true_edges)


def evaluate(true_edges: set, learned_edges: set, all_nodes: list) -> dict:
    """Return shd, fp_rate, fn_rate as a dict."""
    return {
        "shd": compute_shd(true_edges, learned_edges, all_nodes),
        "fp_rate": compute_fp_rate(true_edges, learned_edges, all_nodes),
        "fn_rate": compute_fn_rate(true_edges, learned_edges, all_nodes),
    }
