from __future__ import annotations
import argparse, sys
from bnsl.transforms.shifts import get_shift, shifted_scores
from bnsl.utils.timer import Timer
from bnsl.sampling import sample_data
from bnsl.scoring import write_local_scores, read_local_scores

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

        

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run BNSL algorithms on a .jaa local-scores file")
    ap.add_argument("algorithm", choices=["silander_myllymaki", "partial_order_approach", "approximation_algorithm"],
                    help="Which algorithm to run")
    ap.add_argument("network", type=str, help="Path to the network")
    ap.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate from the network (default: 10000)")
    ap.add_argument("--k", type=int, default=4, help="(approximation_algorithm) number of sets to partition into (default: 4)")
    ap.add_argument("--l", type=int, default=2, help="(approximation_algorithm) parts merged into last bucket (default: 2)")
    ap.add_argument("--m", type=int, default=3, help="(partial_order_approach) size of each bucket (default: 3)")
    ap.add_argument("--p", type=int, default=2, help="(partial_order_approach) number of blocks (default: 2)")
    ap.add_argument("--write_results", type=bool, default=False, help="Whether to write results summary to a file")
    
    args = ap.parse_args(argv)

    # Generate data 
    dat_path = sample_data(args.network, args.num_samples)
    jaa_path = write_local_scores(dat_path)

    LS = read_local_scores(jaa_path)
    timer = Timer()
    timer.start()
    if args.algorithm == "silander_myllymaki":
        from bnsl.algorithms.silander_myllymaki import run
        result = run(jaa_path)
    elif args.algorithm == "partial_order_approach":
        from bnsl.algorithms.partial_order_approach import run
        result = run(jaa_path, m=args.m, p=args.p)
    else:
        from bnsl.algorithms.approximation_algorithm import run
        result = run(jaa_path, l=args.l, k=args.k)

    timer.stop()
    print(f"[{args.algorithm}] elapsed time: {timer.elapsed():.3f} seconds")
    print(f"[{args.algorithm}] score={result.total_score:.3f}")

    if args.algorithm == "approximation_algorithm":
        shift = get_shift(LS)
        shifted_final, shifted_optimal = shifted_scores(shift, len(result.pm), result.total_score, args.l / args.k)
        print(f"[{args.algorithm}] Final Score (shifted back): {shifted_final:.3f}")
        print(f"[{args.algorithm}] Approximation ratio {args.l / args.k} guarantees that the best score is no better than: {shifted_optimal:.3f}")

    print(f"[{args.algorithm}] score={result.total_score:.3f}")
    print(f"[{args.algorithm}] parent map")
    for v in result.pm:
        print(f"  {v}: {result.pm[v]}")

if __name__ == "__main__":
    sys.exit(main())