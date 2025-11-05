import sys
from pathlib import Path
import pytest
from math import comb

ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

from bnsl.algorithms.approximation_algorithm import generate_partial_orders, get_combinations, partition_vertices

def get_expected_size_of_partial_orders(l: int, k: int) -> int:
    """ Returns the expected number of ideals for given m and p values. """
    return comb(k, l)

def get_expected_size_of_sets_W(l: int, k: int) -> int:
    return comb(k, l)

@pytest.mark.parametrize("n, l, k", [(8, 1, 2), (8, 1, 3), (8, 2, 5)])
def test_partial_orders_size(n: int, l: int, k: int):
    """Test that the number of generated partial orders matches the theoretical count."""
    V = [f"{i}" for i in range(n)]
    n = len(V)

    sets = partition_vertices(n, k, V)
    W = get_combinations(sets, l, k)

    assert len(W) == get_expected_size_of_sets_W(l, k), f"Expected {get_expected_size_of_sets_W(l, k)} combinations in W, got {len(W)}"

    count = sum(1 for _ in generate_partial_orders(sets, W))
    expected_size = get_expected_size_of_partial_orders(l, k)

    assert count == expected_size, f"Expected {expected_size} partial orders, got {count} for l={l}, k={k}"