from typing import Dict, Set
from pgmpy.readwrite import BIFReader
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

def compute_shd(network_path: str, pm_learned: Dict[str, Set[str]]) -> int:
    """
    Compute SHD between CPDAGs of true and learned networks using bnlearn in R.
    Args:
        network_path: Path to the BIF file of the true Bayesian network.
        pm_learned: Learned parent mapping, given as a dict
            child -> set of parents.

    Returns:
        int: SHD
    """
    reader = BIFReader(network_path)
    true_model = reader.get_model()
    true_edges = list(true_model.edges()) 

    learned_edges = [(p, c) for c, parents in pm_learned.items() for p in parents]

    bnlearn = rpackages.importr("bnlearn")
    base = rpackages.importr("base")

    nodes = sorted({x for (x, _) in true_edges} | {y for (_, y) in true_edges})

    ro.globalenv["true_nodes"] = ro.StrVector(nodes)
    ro.globalenv["true_from"] = ro.StrVector([u for u, v in true_edges])
    ro.globalenv["true_to"]   = ro.StrVector([v for u, v in true_edges])

    ro.globalenv["learn_from"] = ro.StrVector([u for u, v in learned_edges])
    ro.globalenv["learn_to"]   = ro.StrVector([v for u, v in learned_edges])

    # R code 
    shd_value = ro.r("""
        true_dag <- empty.graph(true_nodes)
        arcs(true_dag) <- data.frame(from=true_from, to=true_to)

        learned_dag <- empty.graph(true_nodes)
        arcs(learned_dag) <- data.frame(from=learn_from, to=learn_to)

        # CPDAG-based SHD (default behaviour)
        shd(true_dag, learned_dag)
    """)

    return int(shd_value[0])

