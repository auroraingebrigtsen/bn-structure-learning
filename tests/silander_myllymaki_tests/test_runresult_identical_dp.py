import sys
from pathlib import Path
import pytest
from bnsl.sampling import sample_data
from bnsl.scoring import write_local_scores
from pgmpy.estimators import BIC
import pandas as pd
from bnsl.algorithms.silander_myllymaki import run
from pygobnilp.gobnilp import Gobnilp


ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

def _pm_to_model(pm):
    from pgmpy.models import DiscreteBayesianNetwork 
    edges = [(p, c) for c, parents in pm.items() for p in parents]
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(pm.keys())
    return model

# Collect all networks in networks/small:
networks = ("asia", "cancer", "survey", "earthquake")
network_paths = [ROOT / "networks" / "small" / f"{net}.bif" for net in networks]

@pytest.mark.slow
@pytest.mark.parametrize("network_path", network_paths)
def test_runresult_match_gobnilp_reference(network_path):
    dat_path = sample_data(network_path, n_samples=100000, seed=42)
    jaa = write_local_scores(dat_path)

    runresult = run(jaa) 

    g = Gobnilp()
    g.learn(
        data_source=str(dat_path),
        data_type="discrete",
        score="DiscreteBIC",
        plot=False,
    )

    gob_parents = {node: set() for node in g.learned_bn.nodes}
    for parent, child in g.learned_bn.edges:
        gob_parents[child].add(parent)

    df = pd.read_csv(
    dat_path,
    sep=" ",  
    header=0,  # first line has the column names
    skiprows=[1],  # skip the arities line
    )
    scorer = BIC(df)

    gob_score = scorer.score(g.learned_bn)
    model = _pm_to_model(runresult.pm)
    score = scorer.score(model)

    if runresult.pm != gob_parents or score != pytest.approx(gob_score):
        print("Discrepancy found:")
        print("Custom parent map:", runresult.pm)
        print("Gobnilp parent map:", gob_parents)
        print("Custom score:", score)
        print("Gobnilp score:", gob_score)

    assert runresult.pm == gob_parents, "Custom parent sets do not match Gobnilp's parent sets"
    assert score == pytest.approx(gob_score), "Custom score does not match Gobnilp's BIC score"
