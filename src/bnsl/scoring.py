import os
from pygobnilp.gobnilp import read_local_scores as gob_read_local_scores, Gobnilp
from pgmpy.readwrite import BIFReader
from typing import Dict, Set

def write_local_scores(dat_path: str) -> str:
    """Write local scores to a file using pygobnilp.
    dat_path: Path to the .dat file containing the data.
    returns: Path to the generated .jaa local scores file.
    """

    file_name = os.path.splitext(os.path.basename(dat_path))[0]
    write_path = f"data/local_scores/{file_name}.jaa"

    if os.path.exists(write_path):
        print(f"File already exists: {write_path}. Skipped computing local scores.")
        return write_path

    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    # Use Gobnilp to read the .dat file and write local scores
    g = Gobnilp()
    g.learn(
        data_source=dat_path,
        data_type="discrete",
        score="DiscreteBIC",
        end="local scores"  # end after computing local scores
    )
    g.write_local_scores(write_path)

    return write_path


def read_local_scores(jaa_path: str) -> dict:
    """Read local scores wrapper from a .jaa file using pygobnilp.
    jaa_path: Path to the .jaa local scores file.
    returns: Local scores dict.
    """
    return gob_read_local_scores(jaa_path)

def _pm_to_edges(pm: Dict[str, Set[str]]) -> set[tuple[str, str]]:
    """Convert child -> parents map into a set of (parent, child) edges."""
    return {(p, c) for c, parents in pm.items() for p in parents}


def compute_shd(network_path: str, pm_learned: Dict[str, Set[str]]) -> int:
    """
    Compute a simple structural Hamming distance between:
      - the true BN in `network_path` (.bif)
      - a learned parent map `pm_learned: child -> set(parents)`

    SHD = #missing edges + #extra edges.
    Edge reversals count as 2 (delete wrong direction, add correct direction).
    """
    reader = BIFReader(network_path)
    true_model = reader.get_model()

    true_edges = set(true_model.edges()) 
    learned_edges = _pm_to_edges(pm_learned)

    missing = true_edges - learned_edges   # in true, not in learned
    extra   = learned_edges - true_edges  # in learned, not in true

    shd = len(missing) + len(extra)
    return shd