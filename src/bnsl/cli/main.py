from __future__ import annotations
import argparse, sys
from bnsl.utils.timer import Timer
from bnsl.sampling import sample_data
from bnsl.transforms.shifts import get_shift, get_upper_bound
from bnsl.scoring import write_local_scores, read_local_scores
from pathlib import Path
import yaml
import json

def _print_current(algorithm: str, network: str, num_samples: int, **kwargs) -> None:
    """Print the current experiment configuration to the console."""
    print(f"\n-------------------------------------")
    print(f"Running {algorithm}:")
    print(f"-------------------------------------")
    print(f"Network: {network}")
    print(f"Num samples: {num_samples}")
    print(f"Parameters: {kwargs}")
    

def _write_results_summary(
    algorithm: str,
    network: str,
    num_samples: int,
    seed: int,
    seconds: float,
    score: float,
    pm: dict[str, set[str]],
    **kwargs) -> None:
    """Write experiment metadata to a .json file inside data/results/."""

    write_dir = Path(f"data/results/{algorithm}/")
    write_dir.mkdir(parents=True, exist_ok=True)

    # Build filename based on algorithm and parameters
    if algorithm == "partial_order_approach":
        file_name = f"{Path(network).stem}_m_{kwargs.get('m')}_p_{kwargs.get('p')}_{num_samples}_seed_{seed}_results.json"
    elif algorithm == "approximation_algorithm":
        file_name = f"{Path(network).stem}_k_{kwargs.get('k')}_l_{kwargs.get('l')}_{num_samples}_seed_{seed}_results.json"
    else:
        file_name = f"{Path(network).stem}_{num_samples}_seed_{seed}_results.json"
    # Convert frozensets to sorted lists for JSON serialization
    parent_map = {node: sorted(list(parents)) for node, parents in pm.items()}

    # Build a structured result object
    result_data = {
        "network": network,
        "algorithm": algorithm,
        "num_variables": len(pm),
        "num_samples": num_samples,
        "seed": seed,
        "seconds_elapsed": round(seconds, 3),
        "score": round(score, 3),
        "params": kwargs,
        "parent_map": parent_map,
    }

    # Write to JSON file
    output_path = write_dir / file_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, sort_keys=True)

    print(f"[{algorithm}] Results written to {output_path}")

def _single_run(algorithm: str, network: str, num_samples: int,  write_results: bool, seed: int, **algo_kwargs) -> None:
    """Run a single experiment with the specified parameters."""
    # Generate data 
    dat_path = sample_data(network, num_samples, seed=seed)
    jaa_path = write_local_scores(dat_path)

    kwargs = {}

    LS = read_local_scores(jaa_path)
    timer = Timer()
    timer.start()
    if algorithm == "silander_myllymaki":
        from bnsl.algorithms.silander_myllymaki import run
        result = run(jaa_path)
    elif algorithm == "partial_order_approach": 
        from bnsl.algorithms.partial_order_approach import run
        result = run(jaa_path, m=algo_kwargs.get("m"), p=algo_kwargs.get("p"))
        kwargs.update({"m": algo_kwargs.get("m"), "p": algo_kwargs.get("p")})
    else:
        from bnsl.algorithms.approximation_algorithm import run
        result = run(jaa_path, l=algo_kwargs.get("l"), k=algo_kwargs.get("k"))
        kwargs.update({"l": algo_kwargs.get("l"), "k": algo_kwargs.get("k")})
        
    if algorithm == "approximation_algorithm":
        shift = get_shift(LS)
        optimal_upper_bound = get_upper_bound(shift, len(result.pm), result.total_score, algo_kwargs.get("k") / algo_kwargs.get("l"))
        kwargs.update({"optimal_upper_bound": optimal_upper_bound})

    timer.stop()
    print(f"[{algorithm}] elapsed time: {timer.elapsed():.3f} seconds")
    print(f"[{algorithm}] score={result.total_score:.3f}")

    if write_results:
        _write_results_summary(
            algorithm=algorithm,
            network=network,
            num_samples=num_samples,
            seed=seed,
            seconds=timer.elapsed(),
            score=result.total_score,
            pm=result.pm,
            **kwargs
        )



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run BNSL algorithms on a .jaa local-scores file")
    ap.add_argument("config", type=str, help="Path to the configuration file (YAML)")
    ap.add_argument("--write_results", action="store_true", help="Whether to write results summary to a file")
    ap.add_argument("--verbose", action="store_true", help="Whether to print current experiment configuration")
    
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    assert "algorithm" in cfg, "Configuration file must specify 'algorithm'"
    assert "networks" in cfg or "networks_dir" in cfg, "Configuration file must specify 'networks' or 'networks_dir'"

    networks = []
    if "networks" in cfg:
        networks = cfg["networks"]
    else:
        networks_dir = cfg["networks_dir"]
        p = Path(networks_dir)
        networks.extend([str(f) for f in p.glob("*.bif")])
    
    for seed in cfg.get("seed", [42]):
        for network in networks:
            for num_samples in cfg.get("sample_sizes", [10000]):
                if cfg["algorithm"] == "approximation_algorithm":
                    for param_set in cfg.get("k_l_grid", [{"k":4, "l":2}]):
                        if args.verbose:
                            _print_current(
                                algorithm=cfg["algorithm"],
                                network=network,
                                num_samples=num_samples,
                                **param_set
                            )
                        _single_run(
                            algorithm=cfg["algorithm"],
                            network=network,
                            num_samples=num_samples,
                            write_results=args.write_results,
                            seed=seed,
                            **param_set
                        )
                elif cfg["algorithm"] == "partial_order_approach":
                    for param_set in cfg.get("m_p_grid", [{"m":3, "p":2}]):
                        if args.verbose:
                            _print_current(
                                algorithm=cfg["algorithm"],
                                network=network,
                                num_samples=num_samples,
                                **param_set
                            )
                        _single_run(
                            algorithm=cfg["algorithm"],
                            network=network,
                            num_samples=num_samples,
                            write_results=args.write_results,
                            seed=seed,
                            **param_set
                        )
                elif cfg["algorithm"] == "silander_myllymaki":
                    if args.verbose:
                        _print_current(
                            algorithm=cfg["algorithm"],
                            network=network,
                            num_samples=num_samples
                        )
                    _single_run(
                        algorithm=cfg["algorithm"],
                        network=network,
                        num_samples=num_samples,
                        seed=seed,
                        write_results=args.write_results
                    )
                else:
                    raise ValueError(f"Unknown algorithm: {cfg['algorithm']}")

if __name__ == "__main__":
    sys.exit(main())