
import sys
from pathlib import Path
import pytest
import math

ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

from bnsl.algorithms.approximation_algorithm import partition_vertices, get_combinations, generate_partial_orders
from bnsl.algorithms.partial_order_approach import predecessors, get_ideals

def get_expected_size_of_ideals(n:int, l: int, k: int) -> int:
    """ Returns the expected number of ideals for a partial order, given l and k values.
     Based on lemma 19 from Partial Order Approach paper: |I(B)| = 1−ℓ+2**|B1|+2**|B2|+···+2**|Bℓ|"""
    
    q, r = divmod(n, k) #partition n into k equally sized sets:  r sets of size q+1, (k-r) sets  of size q

    if l <= r:
        size_big = l * (q + 1)

        num_small_q_plus_1 = r - l 
        num_small_q = k - r  
        L = num_small_q_plus_1 + num_small_q + 1

        return (1- L+ num_small_q_plus_1 * (2 ** (q + 1)) + num_small_q * (2 ** q) + (2 ** size_big))
    else: # l > r
        size_big = r * (q + 1) + (l - r) * q

        num_small_q = k - l 
        L = num_small_q + 1

        return (1- L+ num_small_q * (2 ** q) + (2 ** size_big))

@pytest.mark.parametrize("n, l, k", [(15, 5, 6), (15, 3, 6), (20, 2, 4), (20, 4, 5), (25, 5, 6), (30, 3, 6)])
def test_ideals_size(n: int, l: int, k: int):
    """Test that the number of generated ideals matches the theoretical count."""
    V = [f"{i}" for i in range(n)]
    n = len(V)
    M = set(V)

    sets = partition_vertices(n, k, V)
    W = get_combinations(sets, l, k)

    # pick one partial order P 
    P = generate_partial_orders(sets, W)
    P = next(iter(P))

    # compute ideals
    pred = predecessors(M, P)
    ideals = get_ideals(M, pred)

    # expected count
    expected = get_expected_size_of_ideals(n, l, k)
    assert len(ideals) == expected, f"Expected {expected}, got {len(ideals)} for n={n}, l={l}, k={k}"