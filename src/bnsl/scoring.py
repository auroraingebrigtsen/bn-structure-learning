import os
from pygobnilp.gobnilp import Gobnilp


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
    g = Gobnilp()
    LS = g.read_local_scores(jaa_path)
    return LS