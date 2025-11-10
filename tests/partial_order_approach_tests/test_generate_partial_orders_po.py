import pytest
from math import comb
from bnsl.algorithms.partial_order_approach import generate_partial_orders, make_blocks_and_fronts

def get_expected_size_of_partial_orders(m: int, p: int) -> int:
    """ Returns the expected number of ideals for given m and p values. """
    return comb(m, m // 2) ** p

@pytest.mark.parametrize("n, m, p", [(8, 8, 1), (8, 4, 2), (8, 3, 2), (20, 5, 4), (20, 10, 2)])
def test_partial_orders_size(n: int, m: int, p: int):
    """Test that the number of generated partial orders matches the theoretical count."""
    V = [f"{i}" for i in range(n)]

    blocks, front_choices_per_block, _ = make_blocks_and_fronts(V, m, p)

    count = sum(1 for _ in generate_partial_orders(blocks, front_choices_per_block))
    expected_size = get_expected_size_of_partial_orders(m, p)

    assert count == expected_size, f"Expected {expected_size} partial orders, got {count} for m={m}, p={p}"