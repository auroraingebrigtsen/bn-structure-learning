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

def shifted_scores(shift: float, n: int, raw_score: float, approximation_ratio: float) -> tuple[float, float]:
    """
    Function to compute the final score after shifting back by removing the shift
    contributions from each variable.
    """
    shifted_final = raw_score - (n * shift)
    shifted_optimal = raw_score * approximation_ratio - (n * shift)
    return shifted_final, shifted_optimal
