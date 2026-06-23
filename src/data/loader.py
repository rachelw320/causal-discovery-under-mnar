import numpy as np
import pandas as pd
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
import config


def load_network(name: str):
    """Return a pgmpy BayesianNetwork for 'asia' or 'sachs'."""
    return get_example_model(name)


def sample_data(network, n: int = config.SAMPLE_SIZE, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Sample n rows from a pgmpy BayesianNetwork."""
    sampler = BayesianModelSampling(network)
    df = sampler.forward_sample(size=n, seed=seed)
    return df


def get_true_edges(network) -> set:
    """Return the ground-truth directed edges as a set of (parent, child) tuples."""
    return set(network.edges())
