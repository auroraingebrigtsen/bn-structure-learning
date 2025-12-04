import sys
from pathlib import Path
import pytest
from bnsl.algorithms.silander_myllymaki import run as run_sm
from bnsl.algorithms.approximation_algorithm import run as run_approx
from bnsl.sampling import sample_data
from bnsl.scoring import write_local_scores

ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))

network_paths = list((ROOT / "networks" / "small").glob("*.bif"))

@pytest.mark.slow
@pytest.mark.parametrize("network_path", network_paths)
def test_approx_1_equals_dp(network_path):
    dat = sample_data(network_path, n_samples=10000, seed=42)
    jaa = write_local_scores(dat)
    runresult_dp = run_sm(jaa)
    runresult_approx = run_approx(jaa, k=2, l=2) 

    assert runresult_dp.pm == runresult_approx.pm, f"Parent maps differ for network {network_path}"
    assert runresult_dp.total_score == pytest.approx(runresult_approx.total_score), f"Scores differ for network {network_path}"