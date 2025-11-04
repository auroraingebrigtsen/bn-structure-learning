from __future__ import annotations
import argparse, json, sys
from pygobnilp.gobnilp import read_local_scores
from bnsl.transforms.shifts import get_shift, shifted_scores

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run BNSL algorithms on a .jaa local-scores file")
    ap.add_argument("algorithm", choices=["silander_myllymaki", "partial_order_approach", "approximation_algorithm"],
                    help="Which algorithm to run")
    ap.add_argument("--jaa", required=True, help="Path to .jaa local-scores file")
    ap.add_argument("--k", type=int, default=4, help="(approximation_algorithm) number of sets to partition into")
    ap.add_argument("--l", type=int, default=2, help="(approximation_algorithm) parts merged into last bucket")
    ap.add_argument("--m", type=int, default=3, help="(partial_order_approach) size of each bucket")
    ap.add_argument("--p", type=int, default=2, help="(partial_order_approach) number of blocks")
    
    args = ap.parse_args(argv)

    LS = read_local_scores(args.jaa)

    if args.algorithm == "silander_myllymaki":
        from bnsl.algorithms.silander_myllymaki import run
        result = run(args.jaa)
    elif args.algorithm == "partial_order_approach":
        from bnsl.algorithms.partial_order_approach import run
        result = run(args.jaa, m=args.m, p=args.p)
    else:
        from bnsl.algorithms.approximation_algorithm import run
        result = run(args.jaa, l=args.l, k=args.k)

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

    return 0

if __name__ == "__main__":
    sys.exit(main())