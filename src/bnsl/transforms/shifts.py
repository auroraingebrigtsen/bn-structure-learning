from typing import Dict, FrozenSet

def get_shift(LS: Dict[str, Dict[FrozenSet[str], float]]):
    """
    Function to compute the lowest local score in LS to use as a shift value
    to ensure all scores are non-negative.
    """
    lowest_score = 0
    for _, scored_parent_sets in LS.items():
        for _, score in scored_parent_sets.items():
            if score < lowest_score:
                lowest_score = score
    return -lowest_score

def get_upper_bound(shift: float, n: int, raw_score: float, alpha: float) -> float:
    """
    Function to compute the theoretical optimal upper bound 
    Alpha= = l/k
    """
    g_alg = raw_score + n * shift
    g_opt_ub = alpha * g_alg
    return g_opt_ub - n * shift