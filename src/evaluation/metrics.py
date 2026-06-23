def compute_shd(true_edges: set, learned_edges: set, all_nodes: list) -> int:
    """
    Structural Hamming Distance (SHD) — primary evaluation metric.

    Counts the minimum number of edge insertions, deletions, or reversals
    needed to transform the learned graph into the true graph.

    Each missing edge = 1, each extra edge = 1, each reversed edge = 2
    (one deletion + one insertion), but we simplify here to symmetric difference
    which is standard for skeleton + orientation comparison.
    """
    missing = true_edges - learned_edges
    extra = learned_edges - true_edges
    return len(missing) + len(extra)


def compute_fp_rate(true_edges: set, learned_edges: set, all_nodes: list) -> float:
    """
    False positive edge rate: proportion of learned edges not in the true graph.
    Adding a spurious edge is harmful in causal inference — it implies a causal
    relationship that does not exist.
    """
    if len(learned_edges) == 0:
        return 0.0
    fp = learned_edges - true_edges
    return len(fp) / len(learned_edges)


def compute_fn_rate(true_edges: set, learned_edges: set, all_nodes: list) -> float:
    """
    False negative edge rate: proportion of true edges not recovered.
    Missing a true edge means a real causal relationship is overlooked.
    """
    if len(true_edges) == 0:
        return 0.0
    fn = true_edges - learned_edges
    return len(fn) / len(true_edges)


def evaluate(true_edges: set, learned_edges: set, all_nodes: list) -> dict:
    """Return all three metrics as a dict."""
    return {
        "shd": compute_shd(true_edges, learned_edges, all_nodes),
        "fp_rate": compute_fp_rate(true_edges, learned_edges, all_nodes),
        "fn_rate": compute_fn_rate(true_edges, learned_edges, all_nodes),
    }
