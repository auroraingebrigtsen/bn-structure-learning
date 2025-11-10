from bnsl.transforms.shifts import shifted_scores, get_shift
from bnsl.utils.timer import Timer
from bnsl.scoring import write_local_scores, read_local_scores
from bnsl.sampling import sample_data
from bnsl.algorithms.approximation_algorithm import run
from pathlib import Path

def write_results_summary(
    network: str,
    l:int,
    k:int,
    num_samples: int,
    seconds: float,
    shifted_score: float,
    shifted_optimal: float,
    pm: dict[str, set[str]],
) -> Path:
    """Write experiment metadata to a .txt file inside results/."""
    write_path = "results/approximation_algorithm/"
    file_name = Path(network).stem
    write_path = Path(write_path)
    write_path.mkdir(parents=True, exist_ok=True)
    write_path = write_path / f"{file_name}_{num_samples}_results.txt"

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(f"Network: {network}\n")
        f.write(f"Number of variables: {len(pm)}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Algorithm: approximation_algorithm\n")
        f.write(f"Parameters: l={l}, k={k}\n")
        f.write(f"Elapsed time (seconds): {seconds:.3f}\n")
        f.write(f"Shifted final score: {shifted_score:.3f}\n")
        f.write(f"Shifted optimal score bound: {shifted_optimal:.3f}\n")
        f.write("Parent map:\n")


def run_experiment(network: str, num_samples: int = 10000, l: int = 2, k: int = 4) -> None:
    """Runs a BNSL experiment on the given network using the approximation algorithm.
    
    network: Path to the Bayesian network in BIF format.
    num_samples: Number of samples to generate from the network (default: 10000).
    l: Number of parts merged into the last bucket (default: 2).
    k: Number of sets to partition into (default: 4).
    kwargs: Additional parameters for the algorithms.
    """
    
    # Generate data 
    dat_path = sample_data(network, num_samples)
    jaa_path = write_local_scores(dat_path)

    LS = read_local_scores(jaa_path)

    timer = Timer()
    timer.start()

    result = run(jaa_path, l=l, k=k)

    timer.stop()
    print(f"[approximation_algorithm] elapsed time: {timer.elapsed():.3f} seconds")
    print(f"[approximation_algorithm] score={result.total_score:.3f}")

    shift = get_shift(LS)
    shifted_final, shifted_optimal = shifted_scores(shift, len(result.pm), result.total_score, l / k)

    write_results_summary(
        network=network,
        l=l,
        k=k,
        num_samples=num_samples,
        seconds=timer.elapsed(),
        shifted_score=shifted_final,
        shifted_optimal=shifted_optimal,
        pm=result.pm,
    )


def batch_experiments():
    """Run a batch of experiments on all networks in the networks/ directory
    with varying parameters.
    """
    networks_dir = Path("networks/")
    network_files = list(networks_dir.glob("*.bif"))
    for network_path in network_files:
        network = str(network_path)

        for k, l in [(4,2), (3,2), (5,2)]:
            for num_samples in [1000, 5000, 10000]:
                run_experiment(network, num_samples=num_samples, l=l, k=k)