
import sys
from pathlib import Path
import pytest
import math

ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

from bnsl.algorithms.partial_order_approach import make_blocks_and_fronts, predecessors, get_ideals, generate_partial_orders

def get_expected_size_of_ideals(n:int, m: int, p: int) -> int:
    """ Returns the expected number of ideals for a partial order, given m and p values. """
    return 2**(n - m*p) * (2 **(math.floor(m/2)) + 2**(math.ceil(m/2)) - 1) ** p

@pytest.mark.parametrize("n, m, p", [(8, 8, 1), (8, 4, 2), (8, 3, 2), (20, 5, 4), (20, 10, 2)])
def test_ideals_size(n: int, m: int, p: int):
    """Test that the number of generated ideals matches the theoretical count."""
    V = [f"{i}" for i in range(n)]
    M = set(V)

    blocks, front_choices_per_block = make_blocks_and_fronts(V, m, p)

    # pick one partial order P 
    P = generate_partial_orders(blocks, front_choices_per_block)
    P = next(iter(P))

    # compute ideals
    pred = predecessors(M, P)
    ideals = get_ideals(M, pred)

    # expected count
    expected = get_expected_size_of_ideals(n, m, p)
    assert len(ideals) == expected, f"Expected {expected}, got {len(ideals)} for n={n}, m={m}, p={p}"