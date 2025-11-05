import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

from bnsl.algorithms.silander_myllymaki import run
from pygobnilp.gobnilp import Gobnilp

DATASETS = [
    ("pygobnilp/data/asia_10000.dat", "data/local_scores/local_scores_asia_10000.jaa"),
    #("pygobnilp/data/child_10000.dat", "data/local_scores/local_scores_child_10000.jaa"),
]

@pytest.mark.slow
@pytest.mark.parametrize("dat_path, local_scores_path", DATASETS)
def test_runresult_match_gobnilp_reference(dat_path, local_scores_path):
    """
    Compare the learned edge set to Gobnilp's own output on the same data.
    """
    runresult = run(path=str(local_scores_path))

    g = Gobnilp()
    g.learn(
        data_source=str(dat_path),
        data_type="discrete",
        score="DiscreteBIC",
        plot=False
    )
    gob_parents = {node: set() for node in g.learned_bn.nodes}

    for parent, child in g.learned_bn.edges:
        gob_parents[child].add(parent)
    
    # find out how to get gobnilp score
    #gob_score = g.learned_bn.score

    #assert runresult.total_score == gob_score, "Custom score does not match Gobnilp's score"
    assert runresult.pm == gob_parents, "Custom parent sets do not match Gobnilp's parent sets"
