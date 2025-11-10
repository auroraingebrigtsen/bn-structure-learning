from __future__ import annotations
import argparse, sys
from bnsl.transforms.shifts import get_shift, shifted_scores
from bnsl.utils.timer import Timer
from bnsl.sampling import sample_data
from bnsl.scoring import write_local_scores, read_local_scores
from pathlib import Path
import yaml

def _write_results_summary(
    algorithm: str,
    network: str,
    num_samples: int,
    seconds: float,
    score: float,
    pm: dict[str, set[str]],
    **kwargs
) -> Path:
    """Write experiment metadata to a .txt file inside results/."""

    write_dir = Path(f"data/results/{algorithm}/")
    write_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == "partial_order_approach":
        file_name = Path(network).stem + f"_m_{kwargs.get('m')}_p_{kwargs.get('p')}_{num_samples}_results.txt"
    elif algorithm == "approximation_algorithm":
        file_name = Path(network).stem + f"_k_{kwargs.get('k')}_l_{kwargs.get('l')}_{num_samples}_results.txt"
    else:
        file_name = Path(network).stem + f"_{num_samples}_results.txt"

    with open(write_dir / file_name, "w", encoding="utf-8") as f:
        f.write(f"Network: {network}\n")
        f.write(f"Number of variables: {len(pm)}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Seconds elapsed: {seconds:.3f}\n")
        f.write(f"Score: {score:.3f}\n")
        f.write(f"Parent map: {pm}\n")
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
    
    print(f"[{algorithm}] Results written to {write_dir / file_name}")

def _single_run(algorithm: str, network: str, num_samples: int,  write_results: bool, **algo_kwargs) -> None:
        # Generate data 
    dat_path = sample_data(network, num_samples)
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

    timer.stop()
    print(f"[{algorithm}] elapsed time: {timer.elapsed():.3f} seconds")
    print(f"[{algorithm}] score={result.total_score:.3f}")

    if algorithm == "approximation_algorithm":
        shift = get_shift(LS)
        score, shifted_optimal = shifted_scores(shift, len(result.pm), result.total_score, algo_kwargs.get("l") / algo_kwargs.get("k"))
        kwargs.update({"shifted_optimal": shifted_optimal})
    else:
        score = result.total_score

    for v in result.pm:
        print(f"  {v}: {result.pm[v]}")

    if write_results:
        _write_results_summary(
            algorithm=algorithm,
            network=network,
            num_samples=num_samples,
            seconds=timer.elapsed(),
            score=score,
            pm=result.pm,
            **kwargs
        )



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run BNSL algorithms on a .jaa local-scores file")
    ap.add_argument("config", type=str, help="Path to the configuration file (YAML)")
    ap.add_argument("--write_results", action="store_true", help="Whether to write results summary to a file")
    
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
    
    for network in networks:
        for num_samples in cfg.get("sample_sizes", [10000]):
            if cfg["algorithm"] == "approximation_algorithm":
                for param_set in cfg.get("k_l_grid", [{"k":4, "l":2}]):
                    _single_run(
                        algorithm=cfg["algorithm"],
                        network=network,
                        num_samples=num_samples,
                        write_results=args.write_results,
                        **param_set
                    )
            elif cfg["algorithm"] == "partial_order_approach":
                for param_set in cfg.get("m_p_grid", [{"m":3, "p":2}]):
                    _single_run(
                        algorithm=cfg["algorithm"],
                        network=network,
                        num_samples=num_samples,
                        write_results=args.write_results,
                        **param_set
                    )
            elif cfg["algorithm"] == "silander_myllymaki":
                _single_run(
                    algorithm=cfg["algorithm"],
                    network=network,
                    num_samples=num_samples,
                    write_results=args.write_results
                )
            else:
                raise ValueError(f"Unknown algorithm: {cfg['algorithm']}")

if __name__ == "__main__":
    sys.exit(main())