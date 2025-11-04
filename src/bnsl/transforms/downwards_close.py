from typing import Dict, FrozenSet
from itertools import combinations

def downwards_close(LS: Dict[str, Dict[FrozenSet[str], float]]
) -> Dict[str, Dict[FrozenSet[str], float]]:
    """
    The algorithm assumes that the local scores are downwards closed. That means that if a parent set Z has a defined score for v,
    then all subsets of Z also have a defined score for v.
    This function ensures that the local scores satisfy this property by adding -inf scores for missing subsets.
    """
    closed = {}
    for v, scored_parent_sets in LS.items():
        base = dict(scored_parent_sets)
        all_ps = set()
        for ps in scored_parent_sets.keys():
            for r in range(len(ps) + 1):
                for subset in combinations(ps, r):
                    all_ps.add(frozenset(subset))
        all_ps.add(frozenset())
        for ps in all_ps:
            if ps not in base:
                base[ps] = float('-inf')
        closed[v] = base
    return closed
