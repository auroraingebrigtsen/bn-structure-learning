"""
Implementation of the algorithm presented in:
Kundu, M., Parviainen, P. and Saurabh, S., 2024. Time–Approximation Trade-Offs for Learning Bayesian Networks.
Proceedings of Machine Learning Research (PMLR). 2024, 246, 486-497.
"""

from typing import Iterable, List, FrozenSet, Set
from pygobnilp.gobnilp import read_local_scores
from itertools import combinations
from bnsl.types import Edge, RunResult
from bnsl.transforms.downwards_close import downwards_close
from bnsl.algorithms.partial_order_approach import algorithm1, reconstruct_parent_map

def generate_partial_orders(
        sets: List[FrozenSet[str]], 
        W: List[FrozenSet[str]]
        ) -> Iterable[Set[Edge]]:
    """
    Generates all partial orders that you get by having each W_i as the last bucket, and 
    the rest of the sets in separate buckets in arbitrary order before it
    """
    for w in W:
        early = [set(s) for s in sets if not s.issubset(w)]  # all the sets that are not in w
        buckets = early + [set(w)]  # 
        P: Set[Edge] = set()

        #  Create edges according to the bucket order
        for i in range(len(buckets)): 
            for j in range(i + 1, len(buckets)):
                for u in buckets[i]:
                    for v in buckets[j]:
                        P.add((u, v))  # edges from earlier buckets to later buckets
        yield P


def partition_vertices(n:int, k:int, V:List[str]) -> List[FrozenSet[str]]:
    """
    Function to partition V into k equally sized sets
    """
    q, r = divmod(n, k)  # r parts of size q+1, (k-r) parts of size q
    sets = []
    idx = 0
    for i in range(k):
        size = q + 1 if i < r else q
        sets.append(frozenset(V[idx: idx + size]))
        idx += size
    return sets

def get_combinations(sets: List[FrozenSet[str]], l:int, k:int) -> List[FrozenSet[str]]:
    """
    Function to create a set of all combinations of l sets from the k sets
    """
    W = []
    for comb in combinations(range(k), l):
        selected_vars = set()
        for i in comb:
            selected_vars.update(sets[i])
        W.append(frozenset(selected_vars))
    return W

def run(local_scores_path: str, l:int, k:int):
    """ Main function to run the moderately exponential time algorithm (Section 3) 
    for Bayesian network structure learning.

    local_scores_path: Path to the local scores file in JAA format.
    l: Number of sets to combine in the last bucket of the partial order.
    k: Total number of sets to partition the variables into.
    returns: A parent map representing the optimal Bayesian network structure found.
    """

    LS_raw= read_local_scores(local_scores_path)
    LS = downwards_close(LS_raw)

    V: List[str] = list(LS.keys())
    M: Set[str] = set(V) 
    n = len(V)

    assert 1 <= l <= k <= n 

    best_score = float('-inf')
    best_run = None

    sets = partition_vertices(n, k, V)
    W = get_combinations(sets, l, k)

    for P in generate_partial_orders(sets, W):
        score, run = algorithm1(M, P, LS)
        if score > best_score:
            best_score = score
            best_run = run

    pm = reconstruct_parent_map(
        V,
        best_run['prev'],
        best_run['bps'],
    )

    return RunResult(pm=pm, total_score=best_score)